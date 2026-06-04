#!/usr/bin/env python3
"""Run the AsyncVLA small action head without ROS.

This script is intended for the Jetson PyTorch Docker container. It talks only
over Zenoh:

  subscribes: camera/img_compressed, vla/actions
  publishes:  vla/action_chunk

The ROS container should run zenoh_action_chunk_bridge to republish
vla/action_chunk as /asyncvla/action_chunk.
"""

import argparse
import importlib.util
import io
import json
import os
import sys
import time
import traceback
import types
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import yaml
from PIL import Image


transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def open_zenoh_session(connect_endpoints=None, listen_endpoints=None, mode='client'):
    """Open a Zenoh session without depending on any ROS package."""
    import zenoh

    config = zenoh.Config()
    connect_endpoints = list(connect_endpoints or [])
    listen_endpoints = list(listen_endpoints or [])

    if mode:
        config.insert_json5('mode', json.dumps(mode))
    config.insert_json5('scouting/multicast/enabled', 'false')
    if connect_endpoints:
        config.insert_json5('connect/endpoints', json.dumps(connect_endpoints))
    if listen_endpoints:
        config.insert_json5('listen/endpoints', json.dumps(listen_endpoints))

    return zenoh.open(config)


def delta_to_pose(delta):
    """Integrate [dx, dy, cos(dtheta), sin(dtheta)] deltas into future poses."""
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
    return np.arctan2(np.sin(angle), np.cos(angle))


class ActionHeadConfig:
    resume: bool = True
    vla_path: str = './AsyncVLA_release'
    resume_step: Optional[int] = 750000
    dataset_config_path: str = './config_nav/dataset_config.yaml'


def define_model(cfg):
    Edge_adapter = load_edge_adapter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError(
            'CUDA is not available. Stop here: the Jetson is using CPU torch, '
            'not a Jetson-compatible GPU PyTorch build.'
        )

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    with open(cfg.dataset_config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    shead = Edge_adapter(
        obs_encoding_size=config['obs_encoding_size'],
        mha_num_attention_heads=config['mha_num_attention_heads'],
        mha_num_attention_layers=config['mha_num_attention_layers'],
        mha_ff_dim_factor=config['mha_ff_dim_factor'],
    )

    checkpoint_path = os.path.join(cfg.vla_path, f'shead--{cfg.resume_step}_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Action-head checkpoint not found: {checkpoint_path}')

    print('Loading shead model from', checkpoint_path, flush=True)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("Detected DDP-style checkpoint, removing 'module.' prefix...", flush=True)
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}

    missing, unexpected = shead.load_state_dict(state_dict, strict=False)
    print('Missing keys:', missing, flush=True)
    print('Unexpected keys:', unexpected, flush=True)

    shead.to(torch.bfloat16).to(device=device).eval()
    return shead, device


def load_edge_adapter():
    """Load Edge_adapter without importing the full AsyncVLA training package.

    A normal `from prismatic.models.small_head import Edge_adapter` import also
    executes prismatic package initializers, which pull in RLDS/dlimp training
    dependencies that are not needed for hardware inference. This direct file
    load keeps the Jetson action-head venv small.
    """
    asyncvla_source = next((path for path in sys.path if os.path.exists(os.path.join(path, 'prismatic'))), None)
    if asyncvla_source is None:
        raise RuntimeError('AsyncVLA source path is not on PYTHONPATH/sys.path')

    constants = types.ModuleType('prismatic.vla.constants')
    constants.ACTION_DIM = 4
    constants.ACTION_TOKEN_BEGIN_IDX = 0
    constants.IGNORE_INDEX = -100
    constants.NUM_ACTIONS_CHUNK = 8
    constants.STOP_INDEX = 2

    prismatic_pkg = types.ModuleType('prismatic')
    prismatic_pkg.__path__ = [os.path.join(asyncvla_source, 'prismatic')]
    vla_pkg = types.ModuleType('prismatic.vla')
    vla_pkg.__path__ = [os.path.join(asyncvla_source, 'prismatic', 'vla')]

    sys.modules.setdefault('prismatic', prismatic_pkg)
    sys.modules.setdefault('prismatic.vla', vla_pkg)
    sys.modules['prismatic.vla.constants'] = constants

    small_head_path = os.path.join(asyncvla_source, 'prismatic', 'models', 'small_head.py')
    spec = importlib.util.spec_from_file_location('asyncvla_small_head_runtime', small_head_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Edge_adapter


class ActionHeadHandler:
    def __init__(self, shead, device, action_chunk_pub, metric_waypoint_spacing):
        self.shead = shead
        self.device = device
        self.action_chunk_pub = action_chunk_pub
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.curr_actions = None
        self.past_img = None
        self.curr_img = None
        self.seq_num = 1

    def action_callback(self, msg):
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            dtype = np.dtype(payload.get('dtype', 'float32'))
            shape = tuple(payload['shape'])
            data = np.array(payload['data'], dtype=dtype).reshape(shape)
            self.curr_actions = torch.from_numpy(data).to(self.device).to(torch.bfloat16)
            self.maybe_run()
        except Exception:
            traceback.print_exc()

    def process_image(self, img_bytes):
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = TF.to_tensor(img)
        processed_tensor = TF.resize(img_tensor, (96, 96)).unsqueeze(0)
        return transform(processed_tensor).to(self.device).to(torch.bfloat16)

    def image_callback(self, msg):
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            self.past_img = self.process_image(payload['past_img'].encode('latin-1'))
            self.curr_img = self.process_image(payload['curr_img'].encode('latin-1'))
            self.maybe_run()
        except Exception:
            traceback.print_exc()

    def maybe_run(self):
        if self.curr_actions is None or self.past_img is None or self.curr_img is None:
            return

        start = time.time()
        with torch.no_grad():
            predicted_dactions = self.shead(self.curr_img, self.past_img, self.curr_actions)
            predicted_actions = delta_to_pose(predicted_dactions)

        waypoints = predicted_actions.float().cpu().numpy()
        relative_poses = []
        for x, y, cos_theta, sin_theta in waypoints[0]:
            relative_poses.append([
                float(x * self.metric_waypoint_spacing),
                float(y * self.metric_waypoint_spacing),
                float(clip_angle(np.arctan2(sin_theta, cos_theta))),
            ])

        payload = {
            't_action_head': time.time(),
            'action_head_latency_sec': time.time() - start,
            'frame': 'robot',
            'seq_num': self.seq_num,
            'relative_poses': relative_poses,
        }
        self.action_chunk_pub.put(json.dumps(payload).encode('utf-8'))
        print(
            f'action_chunk #{self.seq_num} sent: {len(relative_poses)} poses '
            f'in {payload["action_head_latency_sec"]:.3f}s',
            flush=True,
        )
        self.seq_num += 1


def parse_args():
    parser = argparse.ArgumentParser(description='Run AsyncVLA small action head without ROS.')
    parser.add_argument('--connect', action='append', default=[], help='Zenoh endpoint, e.g. tcp/BREV_IP:7447')
    parser.add_argument('--listen', action='append', default=[], help='Optional local Zenoh listen endpoint.')
    parser.add_argument('--vla-path', default='./AsyncVLA_release')
    parser.add_argument('--asyncvla-source', default='.')
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
    return parser.parse_args()


def main():
    args = parse_args()
    args.vla_path = os.path.expanduser(args.vla_path)
    args.asyncvla_source = os.path.expanduser(args.asyncvla_source)
    args.dataset_config = os.path.expanduser(args.dataset_config)

    if args.asyncvla_source and args.asyncvla_source not in sys.path:
        sys.path.append(args.asyncvla_source)
    for path in args.extra_python_path:
        path = os.path.expanduser(path)
        if path and not os.path.isabs(path):
            path = os.path.normpath(os.path.join(args.asyncvla_source, path))
        if path and path not in sys.path:
            sys.path.append(path)

    print('Checking CUDA before loading action head...', flush=True)
    print('torch:', torch.__version__, flush=True)
    print('cuda:', torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0), flush=True)

    cfg = ActionHeadConfig()
    cfg.vla_path = args.vla_path
    cfg.dataset_config_path = args.dataset_config
    cfg.resume_step = args.resume_step
    shead, device = define_model(cfg)

    connect_endpoints = args.connect or ['tcp/127.0.0.1:7447']
    with open_zenoh_session(connect_endpoints, args.listen) as z_session:
        action_chunk_pub = z_session.declare_publisher(args.action_chunk_key)
        handler = ActionHeadHandler(
            shead=shead,
            device=device,
            action_chunk_pub=action_chunk_pub,
            metric_waypoint_spacing=args.metric_waypoint_spacing,
        )
        action_sub = z_session.declare_subscriber(args.actions_key, handler.action_callback)
        image_sub = z_session.declare_subscriber(args.camera_key, handler.image_callback)
        print(
            f'Plain action head running on {device}; '
            f'subscribing actions={args.actions_key}, camera={args.camera_key}; '
            f'publishing action_chunk={args.action_chunk_key}',
            flush=True,
        )
        try:
            while True:
                time.sleep(1)
        finally:
            del action_sub
            del image_sub


if __name__ == '__main__':
    main()
