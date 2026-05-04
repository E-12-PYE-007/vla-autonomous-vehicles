# ASClinic VLA Split Architecture Guide

This guide explains how the ASClinic robot stack is intended to run with the slow VLA inference on a cloud GPU and the fast action head on the Jetson Orin Nano.

## 1. System Overview

The project is split into two compute sides:

```text
Cloud GPU:
  run_vla.py
  heavy VLA model
  language + image -> hidden/projected action states

Jetson Orin Nano on robot:
  ROS 2 ASClinic stack
  camera capture
  Zenoh bridges
  fast action head
  wheel PID
  Roboclaw motor driver
```

The reason for the split is that the full VLA model is too heavy for the robot, but the smaller action head can run locally at a faster control rate.

## 2. Runtime Data Flow

The split runtime path is:

```text
Robot camera
  -> /cam
  -> zenoh_camera_bridge
  -> Zenoh key: camera/img_compressed

Robot goal/instruction
  -> /asyncvla/goal
  -> zenoh_instruction_bridge
  -> Zenoh key: robot/goal

Cloud VLA
  subscribes: camera/img_compressed
  subscribes: robot/goal
  publishes:  vla/actions

Jetson action head
  subscribes: vla/actions
  subscribes: camera/img_compressed
  publishes:  vla/action_chunk

Robot control
  vla/action_chunk
  -> zenoh_action_chunk_bridge
  -> /asyncvla/action_chunk
  -> odom_action_chunk_tracker
  -> wheel_velocity_reference
  -> wheel_pid_controller
  -> set_motor_duty_cycle
  -> roboclaw_for_motors
  -> motors
```

Encoder odometry is used to correct trajectory tracking error.

## 3. Main Files

ROS packages:

```text
src/asclinic_vla
src/asclinic_vla_interfaces
```

Important Jetson ROS nodes:

```text
asclinic_camera_capture
goal_publisher
zenoh_camera_bridge
zenoh_instruction_bridge
zenoh_action_chunk_bridge
encoder_odometry
odom_action_chunk_tracker
wheel_pid_controller
roboclaw_for_motors
split_action_head
```

Important launch/config files:

```text
src/asclinic_vla/launch/asclinic_vla_split.launch.py
src/asclinic_vla/config/asclinic_vla_split.yaml
```

Jetson upload bundle:

```text
jetson_upload_bundle/
```

## 4. Preparing The Upload Bundle

The bundle contains the ROS packages and the VLA source needed by the Jetson action head:

```text
jetson_upload_bundle/
  ros2_src/
    asclinic_vla/
    asclinic_vla_interfaces/

  external_runtime/
    AsyncVLA/
    AsyncVLA_release/

  install_on_jetson.sh
  run_split_ros.sh
  run_action_head.sh
  README_JETSON_UPLOAD.md
```

Before uploading, make sure this file exists:

```text
jetson_upload_bundle/external_runtime/AsyncVLA_release/shead--750000_checkpoint.pt
```

If it is missing, place the checkpoint there manually. The bundle currently includes a placeholder:

```text
jetson_upload_bundle/external_runtime/AsyncVLA_release/PUT_CHECKPOINT_HERE.txt
```

## 5. Uploading To The Jetson

From the development machine:

```bash
scp -r jetson_upload_bundle jetson@JETSON_IP:~/
```

Replace:

```text
jetson     -> Jetson username
JETSON_IP  -> Jetson IP address
```

Find the Jetson IP by running this on the Jetson:

```bash
hostname -I
```

## 6. Installing On The Jetson

SSH into the Jetson:

```bash
ssh jetson@JETSON_IP
```

Then install the bundle into the Jetson ROS workspace:

```bash
cd ~/jetson_upload_bundle
bash install_on_jetson.sh
```

This copies:

```text
ros2_src/asclinic_vla
ros2_src/asclinic_vla_interfaces
```

into:

```text
~/cap_ros2_jazzy_ws/vla-autonomous-vehicles/src/
```

Then it builds:

```bash
colcon build --packages-select asclinic_vla_interfaces asclinic_vla
```

## 7. Cloud VLA Setup

On the cloud GPU machine, `run_vla.py` must listen on an address reachable from the Jetson.

Change:

```python
z_conf.insert_json5("listen/endpoints", '["tcp/127.0.0.1:7447"]')
```

to:

```python
z_conf.insert_json5("listen/endpoints", '["tcp/0.0.0.0:7447"]')
```

Then run the cloud VLA process:

```bash
python3 run_vla.py
```

The cloud firewall/security group must allow inbound TCP traffic on:

```text
7447
```

## 8. Running The Robot Side

Terminal 1 on the Jetson, start the ROS side:

```bash
cd ~/jetson_upload_bundle
bash run_split_ros.sh CLOUD_IP true text "go to the red cup"
```

Replace:

```text
CLOUD_IP -> cloud GPU IP address
```

The final `true` means Roboclaw dry-run mode is enabled. Keep this enabled at first.

Terminal 2 on the Jetson, start the action head:

```bash
cd ~/jetson_upload_bundle
bash run_action_head.sh CLOUD_IP
```

For an image goal saved on the Jetson:

```bash
cd ~/jetson_upload_bundle
bash run_split_ros.sh CLOUD_IP true image /home/jetson/goal_images/red_cup.png
```

The `GoalSpec` message supports both text and image goals using boolean flags.
The split Zenoh bridge publishes the full goal on `robot/goal` as JSON. Text
and image goals both use this same path.

The image-goal payload format is:

```json
{
  "mode": "image",
  "use_text": false,
  "use_image": true,
  "goal_text": "navigate to the goal image",
  "goal_image": {
    "encoding": "jpeg",
    "data_b64": "..."
  }
}
```

The bundled cloud `run_vla.py` subscribes to `robot/goal`, decodes image goals,
and stores the goal image. The lightweight VLA path still feeds the model with a
language prompt; direct image-goal conditioning requires using the fuller
AsyncVLA inference path that accepts `goal_image_PIL`.

## 9. What Each Jetson Process Does

`run_split_ros.sh` starts:

```text
asclinic_camera_capture
goal_publisher
zenoh_camera_bridge
zenoh_instruction_bridge
zenoh_action_chunk_bridge
encoder_odometry
odom_action_chunk_tracker
wheel_pid_controller
roboclaw_for_motors
```

`run_action_head.sh` starts:

```text
split_action_head
```

The action head loads:

```text
external_runtime/AsyncVLA/prismatic
external_runtime/AsyncVLA/config_nav/dataset_config.yaml
external_runtime/AsyncVLA_release/shead--750000_checkpoint.pt
```

## 10. First Dry-Run Checks

Keep motors disabled:

```bash
bash run_split_ros.sh CLOUD_IP true
```

In separate Jetson terminals, inspect topics:

```bash
ros2 topic list
ros2 topic echo /cam --once
ros2 topic echo /asyncvla/goal
ros2 topic echo /wheel_velocity_reference
ros2 topic echo /set_motor_duty_cycle
```

Expected behavior:

```text
/cam publishes camera images
/asyncvla/goal publishes the text instruction
cloud receives robot/goal
cloud receives camera/img_compressed
cloud publishes vla/actions
action head prints action_chunk sent
/wheel_velocity_reference starts updating
/set_motor_duty_cycle starts updating
```

## 11. Hardware Safety Checks

Before disabling dry-run:

1. Put the robot on blocks so wheels can spin freely.
2. Confirm the Roboclaw USB port in `asclinic_vla_split.yaml`:

   ```text
   /dev/ttyACM0
   ```

3. Confirm `left_motor_multiplier` and `right_motor_multiplier`.
4. Confirm encoder signs match wheel command signs.
5. Confirm duty cycle values are small and reasonable.

Only then run:

```bash
bash run_split_ros.sh CLOUD_IP false
```

## 12. Current Limitations

The current split path supports text and image goals through one Zenoh key:

```text
/asyncvla/goal -> robot/goal
```

The bundled `run_vla.py` decodes image goals from `robot/goal`, but the
lightweight model call still uses the text prompt/fallback text internally.
Direct image-goal conditioning requires switching the cloud inference code to
the fuller AsyncVLA path that accepts `goal_image_PIL`.

The action-head checkpoint is not included in git or the cloned VLA source. It must be copied separately.

The system currently uses encoder odometry, not map-based or CV localization.

Odometry support is used for both the direct `ActionChunk` trajectory path and
the split action-head path:

```text
encoder_counts -> encoder_odometry -> odom_pose2d
/asyncvla/action_chunk + odom_pose2d -> odom_action_chunk_tracker
odom_action_chunk_tracker -> wheel_velocity_reference
```

This lets the trajectory tracker compare the robot's encoder-estimated pose
against the reference path and correct lateral/heading error.

## 13. Next Engineering Steps

Recommended next steps:

1. Confirm the checkpoint loads on the Jetson.
2. Confirm Jetson CUDA/PyTorch is working.
3. Confirm Zenoh connectivity between Jetson and cloud.
4. Run the full stack in Roboclaw dry-run mode.
5. Add a small Zenoh diagnostic subscriber for:

   ```text
   camera/img_compressed
   robot/goal
   vla/actions
   vla/action_chunk
   ```

6. Tune speed limits in:

   ```text
   split_action_head_runner.py
   asclinic_vla_split.yaml
   ```

7. Tune wheel PID gains on the actual robot.
8. Verify motor and encoder sign conventions.
9. Add a launch option for goal text so you do not need to edit YAML.
10. Add logging for latency:

    ```text
    camera timestamp
    cloud VLA timestamp
    action-head timestamp
    wheel command timestamp
    ```

11. Add a watchdog that stops the robot if cloud/Zenoh/action-head updates stall.
12. Once stable, remove old prototype packages:

    ```text
    src/asyncvla_ros
    src/asyncvla_interfaces
    ```

## 14. Quick Command Summary

Upload:

```bash
scp -r jetson_upload_bundle jetson@JETSON_IP:~/
```

Install on Jetson:

```bash
cd ~/jetson_upload_bundle
bash install_on_jetson.sh
```

Run cloud:

```bash
python3 run_vla.py
```

Run Jetson ROS side:

```bash
bash run_split_ros.sh CLOUD_IP true
```

Run Jetson action head:

```bash
bash run_action_head.sh CLOUD_IP
```

Run with motors enabled only after safety checks:

```bash
bash run_split_ros.sh CLOUD_IP false
```
