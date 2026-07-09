#!/usr/bin/env python3
"""Run the fast AsyncVLA action head on the Jetson side of the split system.

The cloud VLA publishes hidden/projected action states on Zenoh `vla/actions`.
This process receives those states plus the current/past camera images, runs the
small edge adapter (`Edge_adapter`) locally, converts its output into a relative
pose trajectory, and publishes `vla/action_chunk` for the ROS trajectory tracker.
"""

import argparse
import io
import json
import os
import sys
import time
import traceback
from typing import Optional, Tuple

import numpy as np
import rclpy
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import yaml
from asclinic_vla_interfaces.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from PIL import Image
from rclpy.node import Node

from asclinic_vla.zenoh_utils import open_zenoh_session


transform = transforms.Compose([
    # Match the image normalization used by the original action-head script.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def delta_to_pose(delta):
    """Integrate action-head delta outputs into a pose chunk.

    The action head emits [dx, dy, cos(dtheta), sin(dtheta)] deltas. This
    function accumulates those deltas into future robot-frame poses represented
    as [x, y, cos(theta), sin(theta)].
    """
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    _, time_steps = dx.shape
    poses = []
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    for idx in range(1, time_steps):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        dx_w = ct * dx[:, idx] - st * dy[:, idx]
        dy_w = st * dx[:, idx] + ct * dy[:, idx]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, idx]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    return torch.stack(poses, dim=1)


def clip_angle(angle):
    """Wrap an angle to [-pi, pi] using atan2 for numerical stability."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class ActionHeadInference:
    """Owns one action-head inference pass for a synchronized action/image pair."""

    def __init__(self, projected_actions, shead, device_id, past_img, curr_img, metric_waypoint_spacing):
        self.projected_actions = projected_actions
        self.shead = shead
        self.device_id = device_id
        self.past_img = past_img
        self.curr_img = curr_img
        self.metric_waypoint_spacing = metric_waypoint_spacing

    def run_shead(self):
        """Run Edge_adapter and return relative poses as [x, y, theta]."""
        with torch.no_grad():
            predicted_dactions = self.shead(self.curr_img, self.past_img, self.projected_actions)
            predicted_actions = delta_to_pose(predicted_dactions)
        return self.pose_chunk_from_actions(predicted_actions.cpu(), self.metric_waypoint_spacing)

    @staticmethod
    def pose_chunk_from_actions(actions, metric_waypoint_spacing):
        """Convert [x, y, cos(theta), sin(theta)] waypoints into metric poses."""
        waypoints = actions.float().cpu().numpy()
        poses = []
        for x, y, cos_theta, sin_theta in waypoints[0]:
            poses.append([
                float(x * metric_waypoint_spacing),
                float(y * metric_waypoint_spacing),
                float(clip_angle(np.arctan2(sin_theta, cos_theta))),
            ])
        return poses


class ActionHeadConfig:
    """Minimal model-loading configuration for the Jetson action-head runner."""

    resume: bool = True
    vla_path: str = './AsyncVLA_release'
    resume_step: Optional[int] = 750000
    dataset_config_path: str = './config_nav/dataset_config.yaml'


def define_model(cfg: ActionHeadConfig):
    """Build Edge_adapter and load the shead checkpoint."""
    from prismatic.models.small_head import Edge_adapter

    # Prefer CUDA on the Jetson Orin Nano; CPU fallback is only for debugging.
    device_id = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

    with open(cfg.dataset_config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Edge_adapter architecture must match the training config, so all dimensions
    # are read from dataset_config.yaml instead of being hard-coded here.
    shead = Edge_adapter(
        obs_encoding_size=config['obs_encoding_size'],
        mha_num_attention_heads=config['mha_num_attention_heads'],
        mha_num_attention_layers=config['mha_num_attention_layers'],
        mha_ff_dim_factor=config['mha_ff_dim_factor'],
    )

    checkpoint_path = os.path.join(cfg.vla_path, f'shead--{cfg.resume_step}_checkpoint.pt')
    if cfg.resume and os.path.exists(checkpoint_path):
        print('Loading shead model from', checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if any(key.startswith('module.') for key in state_dict.keys()):
            # Training checkpoints saved under DistributedDataParallel often
            # prefix parameters with `module.`; remove it for single-GPU inference.
            state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
        missing, unexpected = shead.load_state_dict(state_dict, strict=False)
        print('Missing keys:', missing)
        print('Unexpected keys:', unexpected)
    elif cfg.resume:
        raise FileNotFoundError(f'Action-head checkpoint not found: {checkpoint_path}')

    shead.to(torch.bfloat16).to(device=device_id).eval()
    return shead, device_id


class RosActionChunkPublisher:
    """Publishes action-head pose chunks directly into the local ROS graph."""

    def __init__(self, action_topic):
        self.node = Node('split_action_head_ros_publisher')
        self.publisher = self.node.create_publisher(ActionChunk, action_topic, 10)
        self.node.get_logger().info(f'Publishing local ROS ActionChunk on {action_topic}')

    def publish(self, payload):
        msg = ActionChunk()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = payload.get('frame', 'robot')
        msg.seq_num = int(payload['seq_num'])

        for item in payload['relative_poses']:
            pose = Pose2D()
            pose.x = float(item[0])
            pose.y = float(item[1])
            pose.theta = float(item[2])
            msg.relative_poses.append(pose)

        self.publisher.publish(msg)

    def destroy(self):
        self.node.destroy_node()


class ActionHeadZenohHandler:
    """Stores latest Zenoh inputs; inference is driven by an external 3 Hz timer."""

    def __init__(self, shead, device_id, action_chunk_pub, ros_action_chunk_pub, metric_waypoint_spacing):
        self.shead = shead
        self.device_id = device_id
        self.curr_actions = None
        self.past_frame_id = None
        self.curr_img = None
        self._frame_buffer = {}
        self._frame_buffer_maxsize = 30
        self.action_chunk_pub = action_chunk_pub
        self.ros_action_chunk_pub = ros_action_chunk_pub
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.seq_num = 1

    def action_callback(self, msg):
        """Decode cloud VLA projected actions and record which frame they used."""
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            dtype = np.dtype(payload.get('dtype', 'float32'))
            shape = tuple(payload['shape'])
            data = np.array(payload['data'], dtype=dtype).reshape(shape)
            self.curr_actions = torch.from_numpy(data).to(self.device_id).to(torch.bfloat16)
            self.past_frame_id = payload.get('frame_id')
        except Exception:
            traceback.print_exc()

    def process_image(self, img_bytes):
        """Decode one JPEG frame and preprocess it for Edge_adapter."""
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = TF.to_tensor(img)
        h, w = img_tensor.shape[-2], img_tensor.shape[-1]
        img_tensor = TF.center_crop(img_tensor, min(h, w))
        img_tensor = TF.resize(img_tensor, (224, 224))
        processed_tensor = TF.resize(img_tensor, (96, 96)).unsqueeze(0)
        return transform(processed_tensor).to(self.device_id).to(torch.bfloat16)

    def img_callback(self, msg):
        """Store the latest camera frame and add it to the frame buffer."""
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            frame_id = payload.get('frame_id')
            curr = self.process_image(payload['curr_img'].encode('latin-1'))
            if frame_id is not None:
                self._frame_buffer[frame_id] = curr
                if len(self._frame_buffer) > self._frame_buffer_maxsize:
                    del self._frame_buffer[min(self._frame_buffer)]
            self.curr_img = curr
        except Exception:
            traceback.print_exc()

    def run_inference(self):
        """Run one action-head inference step; called by the 3 Hz main loop."""
        if self.curr_actions is None or self.curr_img is None:
            return
        if self.past_frame_id is None:
            return
        past_img = self._frame_buffer.get(self.past_frame_id)
        if past_img is None:
            print(f'[run_inference] frame_id {self.past_frame_id} not in buffer', flush=True)
            return

        inference = ActionHeadInference(
            projected_actions=self.curr_actions,
            shead=self.shead,
            device_id=self.device_id,
            past_img=past_img,
            curr_img=self.curr_img,
            metric_waypoint_spacing=self.metric_waypoint_spacing,
        )
        relative_poses = inference.run_shead()
        payload = {
            't_action_head': time.time(),
            'frame': 'robot',
            'seq_num': self.seq_num,
            'relative_poses': relative_poses,
        }
        if self.action_chunk_pub is not None:
            self.action_chunk_pub.put(json.dumps(payload).encode('utf-8'))
        if self.ros_action_chunk_pub is not None:
            self.ros_action_chunk_pub.publish(payload)
        print(f'action_chunk #{self.seq_num} sent: {len(relative_poses)} poses', flush=True)
        self.seq_num += 1


def parse_args(argv):
    """Parse runtime options so paths and Zenoh endpoints do not need code edits."""
    parser = argparse.ArgumentParser(description='Run AsyncVLA fast action head on the Jetson.')
    parser.add_argument('--connect', action='append', default=[], help='Zenoh endpoint to connect to, e.g. tcp/CLOUD_IP:7447')
    parser.add_argument('--listen', action='append', default=[], help='Zenoh endpoint to listen on, e.g. tcp/0.0.0.0:7448')
    parser.add_argument('--vla-path', default='./AsyncVLA_release')
    parser.add_argument('--asyncvla-source', default='./AsyncVLA', help='Path containing the prismatic Python package.')
    parser.add_argument('--dataset-config', default='./config_nav/dataset_config.yaml')
    parser.add_argument('--resume-step', type=int, default=750000)
    parser.add_argument('--actions-key', default='vla/actions')
    parser.add_argument('--camera-key', default='camera/img_compressed')
    parser.add_argument('--action-chunk-key', default='vla/action_chunk')
    parser.add_argument('--ros-action-topic', default='', help='Also publish ActionChunk locally on this ROS topic.')
    parser.add_argument('--disable-zenoh-action-chunk-publish', action='store_true')
    parser.add_argument('--metric-waypoint-spacing', type=float, default=0.1)
    parser.add_argument('--extra-python-path', action='append', default=[
        '../Learning-to-Drive-Anywhere-with-MBRA/train/',
        '../lerobot',
    ])
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv=None):
    """Entry point used by `ros2 run asclinic_vla split_action_head -- ...`."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.vla_path = os.path.expanduser(args.vla_path)
    args.asyncvla_source = os.path.expanduser(args.asyncvla_source)
    args.dataset_config = os.path.expanduser(args.dataset_config)

    if args.asyncvla_source and args.asyncvla_source not in sys.path:
        sys.path.append(args.asyncvla_source)
    for path in args.extra_python_path:
        # The AsyncVLA repo keeps prismatic/lerobot modules outside this ROS
        # package, so expose those source roots before importing Edge_adapter.
        path = os.path.expanduser(path)
        if path and not os.path.isabs(path):
            path = os.path.normpath(os.path.join(args.asyncvla_source, path))
        if path and path not in sys.path:
            sys.path.append(path)

    cfg = ActionHeadConfig()
    cfg.vla_path = args.vla_path
    cfg.dataset_config_path = args.dataset_config
    cfg.resume_step = args.resume_step

    shead, device_id = define_model(cfg)

    connect_endpoints = args.connect or ['tcp/127.0.0.1:7447']
    ros_action_chunk_pub = None
    if args.ros_action_topic:
        rclpy.init(args=None)
        ros_action_chunk_pub = RosActionChunkPublisher(args.ros_action_topic)

    with open_zenoh_session(connect_endpoints, args.listen) as z_session:
        # The action head subscribes to camera/actions and publishes a relative
        # trajectory that the ROS odometry-aware tracker turns into wheel refs.
        action_chunk_publisher = None
        if not args.disable_zenoh_action_chunk_publish:
            action_chunk_publisher = z_session.declare_publisher(args.action_chunk_key)
        handler = ActionHeadZenohHandler(
            shead,
            device_id,
            action_chunk_publisher,
            ros_action_chunk_pub,
            args.metric_waypoint_spacing,
        )
        action_subscriber = z_session.declare_subscriber(args.actions_key, handler.action_callback)
        img_subscriber = z_session.declare_subscriber(args.camera_key, handler.img_callback)
        print(
            f'Action head running on {device_id}; '
            f'actions={args.actions_key}, camera={args.camera_key}, '
            f'zenoh_action_chunk={args.action_chunk_key if action_chunk_publisher else "disabled"}, '
            f'ros_action_topic={args.ros_action_topic or "disabled"}',
            flush=True,
        )
        interval = 1 / 3
        try:
            while True:
                t0 = time.time()
                if ros_action_chunk_pub is not None:
                    rclpy.spin_once(ros_action_chunk_pub.node, timeout_sec=0.0)
                handler.run_inference()
                time.sleep(max(0.0, interval - (time.time() - t0)))
        finally:
            del action_subscriber
            del img_subscriber
            if ros_action_chunk_pub is not None:
                ros_action_chunk_pub.destroy()
                rclpy.shutdown()


if __name__ == '__main__':
    main()
