#!/usr/bin/env python3
"""Benchmark AsyncVLA's small action head with synthetic inputs.

This script is intentionally independent of ROS and Zenoh. It loads only the
small-head model definition, creates fake current/past images plus fake VLA
action-token features, and reports forward-pass latency. Use it on the Jetson
Orin Nano to measure the fast action head before the cloud VLA is available.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import torch
import yaml


def load_module(module_name, module_path):
    """Load a Python file as a module without importing its parent package."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_minimal_prismatic_modules(asyncvla_source):
    """Expose only the prismatic modules needed by prismatic/models/small_head.py.

    Importing `prismatic.models.small_head` normally executes prismatic package
    initializers that depend on the full VLA stack. For a small-head-only
    benchmark, we avoid those imports and load just constants.py plus
    small_head.py.
    """
    prismatic_dir = os.path.join(asyncvla_source, 'prismatic')
    constants_path = os.path.join(prismatic_dir, 'vla', 'constants.py')

    prismatic_pkg = types.ModuleType('prismatic')
    prismatic_pkg.__path__ = [prismatic_dir]
    sys.modules.setdefault('prismatic', prismatic_pkg)

    vla_pkg = types.ModuleType('prismatic.vla')
    vla_pkg.__path__ = [os.path.join(prismatic_dir, 'vla')]
    sys.modules.setdefault('prismatic.vla', vla_pkg)

    load_module('prismatic.vla.constants', constants_path)


def load_edge_adapter(asyncvla_source):
    """Return the Edge_adapter class from AsyncVLA's small_head.py."""
    visualnav_train = os.path.join(asyncvla_source, 'visualnav-transformer', 'train')
    if visualnav_train not in sys.path:
        sys.path.insert(0, visualnav_train)

    install_minimal_prismatic_modules(asyncvla_source)
    small_head_path = os.path.join(asyncvla_source, 'prismatic', 'models', 'small_head.py')
    small_head = load_module('asyncvla_small_head_for_benchmark', small_head_path)
    return small_head.Edge_adapter


def choose_dtype(dtype_name, device):
    """Choose the requested tensor dtype, with safe CPU fallback."""
    if dtype_name == 'float32':
        return torch.float32
    if dtype_name == 'float16':
        return torch.float16 if device.type == 'cuda' else torch.float32
    return torch.bfloat16 if device.type == 'cuda' else torch.float32


def build_model(args, device, dtype):
    """Construct the action head and optionally load the shead checkpoint."""
    with open(args.dataset_config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    EdgeAdapter = load_edge_adapter(args.asyncvla_source)
    model = EdgeAdapter(
        obs_encoding_size=config['obs_encoding_size'],
        mha_num_attention_heads=config['mha_num_attention_heads'],
        mha_num_attention_layers=config['mha_num_attention_layers'],
        mha_ff_dim_factor=config['mha_ff_dim_factor'],
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if any(key.startswith('module.') for key in checkpoint.keys()):
            checkpoint = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f'checkpoint={args.checkpoint}')
        print(f'missing_keys={len(missing)} unexpected_keys={len(unexpected)}')

    model.to(device=device, dtype=dtype).eval()
    return model, int(config['obs_encoding_size'])


def make_fake_inputs(batch_size, action_tokens, embedding_dim, device, dtype):
    """Create fake small-head inputs with the same shapes as split inference."""
    curr_img = torch.randn(batch_size, 3, 96, 96, device=device, dtype=dtype)
    past_img = torch.randn(batch_size, 3, 96, 96, device=device, dtype=dtype)
    vla_feature = torch.randn(batch_size, action_tokens, embedding_dim, device=device, dtype=dtype)
    return curr_img, past_img, vla_feature


def synchronize_if_needed(device):
    """Wait for CUDA kernels so timing includes the actual GPU work."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


def percentile(values, pct):
    """Compute one latency percentile in milliseconds."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def run_benchmark(args):
    """Load the model, run warmup passes, then report timed inference latency."""
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    dtype = choose_dtype(args.dtype, device)
    model, embedding_dim = build_model(args, device, dtype)
    curr_img, past_img, vla_feature = make_fake_inputs(
        args.batch_size,
        args.action_tokens,
        embedding_dim,
        device,
        dtype,
    )

    print(f'device={device} dtype={dtype} embedding_dim={embedding_dim}')
    print(
        'input_shapes='
        f'curr_img{tuple(curr_img.shape)}, '
        f'past_img{tuple(past_img.shape)}, '
        f'vla_feature{tuple(vla_feature.shape)}'
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(curr_img, past_img, vla_feature)
        synchronize_if_needed(device)

        latencies_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = model(curr_img, past_img, vla_feature)
            synchronize_if_needed(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    summary = {
        'iterations': args.iterations,
        'warmup': args.warmup,
        'batch_size': args.batch_size,
        'output_shape': list(output.shape),
        'mean_ms': float(np.mean(latencies_ms)),
        'median_ms': percentile(latencies_ms, 50),
        'p90_ms': percentile(latencies_ms, 90),
        'p99_ms': percentile(latencies_ms, 99),
        'min_ms': float(np.min(latencies_ms)),
        'max_ms': float(np.max(latencies_ms)),
    }
    if device.type == 'cuda':
        summary['max_cuda_memory_mb'] = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    print(json.dumps(summary, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Benchmark AsyncVLA small action head with fake inputs.')
    parser.add_argument('--asyncvla-source', required=True, help='Path to the AsyncVLA source directory.')
    parser.add_argument('--dataset-config', required=True, help='Path to config_nav/dataset_config.yaml.')
    parser.add_argument('--checkpoint', default='', help='Optional shead checkpoint path.')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--action-tokens', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_benchmark(args)


if __name__ == '__main__':
    main()
