# Jetson Small Action-Head Benchmark

This test measures the AsyncVLA small action head on the Jetson Orin Nano using
fake inputs. It does not require the cloud VLA, Zenoh, camera input, or robot
motors.

## Minimal Files Needed On The Jetson

Keep the payload small by copying only:

```text
jetson_upload_bundle/
  ros2_src/
    asclinic_vla/
    asclinic_vla_interfaces/
  external_runtime/
    AsyncVLA/
      config_nav/dataset_config.yaml
      prismatic/models/small_head.py
      prismatic/vla/constants.py
      visualnav-transformer/train/vint_train/
    AsyncVLA_release/
      shead--750000_checkpoint.pt
  install_on_jetson.sh
  run_action_head_benchmark.sh
```

The checkpoint is optional for pure timing. Runtime is almost identical with
random weights, but using the checkpoint is better for real memory-use testing.

## Python Dependencies

The benchmark avoids importing the full AsyncVLA package. It needs only:

```bash
python3 -m pip install --user \
  numpy \
  pyyaml \
  pillow \
  efficientnet_pytorch
```

For `torch` and `torchvision`, use the versions already installed for your
Jetson/JetPack image when possible. Do not blindly install the desktop PyPI
CUDA wheels on the Jetson.

Check CUDA visibility:

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

## Build The ROS Package

On the Jetson:

```bash
cd ~/jetson_upload_bundle
bash install_on_jetson.sh
```

This installs and builds `asclinic_vla_interfaces` and `asclinic_vla` into:

```text
~/cap_ros2_jazzy_ws/vla-autonomous-vehicles
```

## Run The Benchmark

```bash
cd ~/jetson_upload_bundle
bash run_action_head_benchmark.sh
```

Optional settings:

```bash
ITERATIONS=300 WARMUP=30 DTYPE=bfloat16 bash run_action_head_benchmark.sh
```

To benchmark without a checkpoint, leave
`external_runtime/AsyncVLA_release/shead--750000_checkpoint.pt` absent. The
script will warn and use random weights.

## Fake Inputs Used

The script creates:

```text
curr_img:     [1, 3, 96, 96]
past_img:     [1, 3, 96, 96]
vla_feature:  [1, 8, 1024]
```

These match the current `dataset_config.yaml`:

```text
len_traj_pred: 8
obs_encoding_size: 1024
image_size: [96, 96]
```

## Reading The Output

The final JSON includes:

```text
mean_ms
median_ms
p90_ms
p99_ms
min_ms
max_ms
max_cuda_memory_mb
output_shape
```

`output_shape` should be:

```text
[1, 8, 4]
```

That means the small head produced an 8-step action chunk with four values per
step.
