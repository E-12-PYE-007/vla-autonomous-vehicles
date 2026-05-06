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
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

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


class ActionHeadZenohHandler:
    """Keeps latest Zenoh inputs and triggers action-head inference when ready."""

    def __init__(self, shead, device_id, action_chunk_pub, metric_waypoint_spacing):
        self.shead = shead
        self.device_id = device_id
        self.curr_actions = None
        self.past_img = None
        self.curr_img = None
        self.action_chunk_pub = action_chunk_pub
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.seq_num = 1

    def action_callback(self, msg):
        """Decode cloud VLA projected actions from JSON into a bfloat16 tensor."""
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            dtype = np.dtype(payload.get('dtype', 'float32'))
            shape = tuple(payload['shape'])
            data = np.array(payload['data'], dtype=dtype).reshape(shape)
            self.curr_actions = torch.from_numpy(data).to(self.device_id).to(torch.bfloat16)
            self.maybe_run()
        except Exception:
            traceback.print_exc()

    def process_image(self, img_bytes):
        """Decode one JPEG frame and preprocess it for Edge_adapter."""
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = TF.to_tensor(img)
        processed_tensor = TF.resize(img_tensor, (96, 96)).unsqueeze(0)
        return transform(processed_tensor).to(self.device_id).to(torch.bfloat16)

    def img_callback(self, msg):
        """Decode the previous/current image pair from Zenoh camera JSON."""
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            self.past_img = self.process_image(payload['past_img'].encode('latin-1'))
            self.curr_img = self.process_image(payload['curr_img'].encode('latin-1'))
            self.maybe_run()
        except Exception:
            traceback.print_exc()

    def maybe_run(self):
        """Run the action head once all three inputs are available."""
        if self.curr_actions is None or self.past_img is None or self.curr_img is None:
            return

        inference = ActionHeadInference(
            projected_actions=self.curr_actions,
            shead=self.shead,
            device_id=self.device_id,
            past_img=self.past_img,
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
        self.action_chunk_pub.put(json.dumps(payload).encode('utf-8'))
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
    if args.asyncvla_source and args.asyncvla_source not in sys.path:
        sys.path.append(args.asyncvla_source)
    for path in args.extra_python_path:
        # The AsyncVLA repo keeps prismatic/lerobot modules outside this ROS
        # package, so expose those source roots before importing Edge_adapter.
        if path and path not in sys.path:
            sys.path.append(path)

    cfg = ActionHeadConfig()
    cfg.vla_path = args.vla_path
    cfg.dataset_config_path = args.dataset_config
    cfg.resume_step = args.resume_step

    shead, device_id = define_model(cfg)

    connect_endpoints = args.connect or ['tcp/127.0.0.1:7447']
    with open_zenoh_session(connect_endpoints, args.listen) as z_session:
        # The action head subscribes to camera/actions and publishes a relative
        # trajectory that the ROS odometry-aware tracker turns into wheel refs.
        action_chunk_publisher = z_session.declare_publisher(args.action_chunk_key)
        handler = ActionHeadZenohHandler(
            shead,
            device_id,
            action_chunk_publisher,
            args.metric_waypoint_spacing,
        )
        action_subscriber = z_session.declare_subscriber(args.actions_key, handler.action_callback)
        img_subscriber = z_session.declare_subscriber(args.camera_key, handler.img_callback)
        print(
            f'Action head running on {device_id}; '
            f'actions={args.actions_key}, camera={args.camera_key}, action_chunk={args.action_chunk_key}',
            flush=True,
        )
        try:
            while True:
                time.sleep(1)
        finally:
            del action_subscriber
            del img_subscriber


if __name__ == '__main__':
    main()
