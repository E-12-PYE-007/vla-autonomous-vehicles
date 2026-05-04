# Jetson Upload Bundle

Upload this whole folder to the Jetson, for example:

```bash
scp -r jetson_upload_bundle jetson@JETSON_IP:~/
```

On the Jetson, the expected layout is:

```text
~/jetson_upload_bundle/
  ros2_src/
    asclinic_vla/
    asclinic_vla_interfaces/
  external_runtime/
    AsyncVLA/
    AsyncVLA_release/
```

## Missing Checkpoint

The cloned VLA repo did not contain:

```text
external_runtime/AsyncVLA_release/shead--750000_checkpoint.pt
```

Before running the action head, put that checkpoint file into:

```text
~/jetson_upload_bundle/external_runtime/AsyncVLA_release/shead--750000_checkpoint.pt
```

The dataset config is already included at:

```text
~/jetson_upload_bundle/external_runtime/AsyncVLA/config_nav/dataset_config.yaml
```

## Install ROS Packages On Jetson

From the Jetson:

```bash
cd ~/jetson_upload_bundle
bash install_on_jetson.sh
```

This copies the ROS packages into:

```text
~/cap_ros2_jazzy_ws/vla-autonomous-vehicles/src/
```

Then it builds:

```text
asclinic_vla_interfaces
asclinic_vla
```

## Run The Split ROS Stack

Terminal 1 on the Jetson:

```bash
cd ~/jetson_upload_bundle
bash run_split_ros.sh CLOUD_IP true text "go to the red cup"
```

The final `true` keeps Roboclaw in dry-run mode.

For an image goal saved on the Jetson:

```bash
bash run_split_ros.sh CLOUD_IP true image /home/jetson/goal.png
```

## Run The Jetson Action Head

Terminal 2 on the Jetson:

```bash
cd ~/jetson_upload_bundle
bash run_action_head.sh CLOUD_IP
```

## Cloud VLA Script

On the cloud machine, `run_vla.py` must listen on all interfaces:

```python
z_conf.insert_json5("listen/endpoints", '["tcp/0.0.0.0:7447"]')
```

The bundled copy at:

```text
external_runtime/AsyncVLA/inference/run_vla.py
```

has also been patched to subscribe to `robot/goal`, including image-goal JSON
payloads. It decodes and stores the goal image; the lightweight model path still
uses the text prompt/fallback text for inference.

## Safety

Keep `roboclaw_dry_run:=true` until:

- the cloud VLA receives `robot/goal`
- the cloud VLA publishes `vla/actions`
- the Jetson action head publishes `vla/action_chunk`
- `/asyncvla/action_chunk` looks sane
- `wheel_velocity_reference` looks sane
- motor signs are checked with the robot lifted
