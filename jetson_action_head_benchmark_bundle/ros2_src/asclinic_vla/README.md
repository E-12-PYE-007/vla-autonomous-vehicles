# ASClinic VLA Integration

This package is the workspace for adapting the ASClinic robot stack to the existing AsyncVLA ROS 2 pipeline.

## Intended Runtime Pipeline

1. Camera node publishes robot camera frames for the VLA input stream.
2. Slow VLA inference runs remotely and publishes predicted trajectory/action chunks through HTTP API or Zenoh.
3. Robot-side fast action head/adaptor converts model output into a reference trajectory.
4. Pure pursuit trajectory tracker consumes the reference trajectory and publishes desired body velocities.
5. Wheel PID controller converts desired body velocities into left/right duty-cycle commands.
6. Roboclaw motor node sends duty-cycle commands to the motors.
7. Odometry provides the robot state estimate for trajectory tracking.

## ASClinic Components To Port

- `odometry_node.py`
- `wheel_pid_controller.py`
- `purepursuit_tracker.py`
- `trajectory_tracker.py`
- `roboclaw_for_motors.py`
- `camera_capture.py`
- message definitions for left/right wheel encoder and duty-cycle traffic

CV localisation nodes should stay out of this integration path.

## First Implementation Milestones

1. Tune `config/asclinic_vla_hardware.yaml` for the Jetson camera, Roboclaw serial port, wheel geometry, and VLA API URL.
2. Launch in dry-run mode first:

   ```bash
   ros2 launch asclinic_vla asclinic_vla_hardware.launch.py roboclaw_dry_run:=true
   ```

3. Publish a goal on `/asyncvla/goal`, enable the included goal publisher with `use_goal_publisher:=true`, or disable `use_vla_http_client` and mock `/asyncvla/action_chunk` directly.
4. Confirm the topic chain:

   ```text
   goal_publisher -> /asyncvla/goal
   /cam + /asyncvla/goal -> vla_http_client -> /asyncvla/action_chunk
   /asyncvla/action_chunk -> action_chunk_pure_pursuit -> wheel_velocity_reference
   wheel_velocity_reference + encoder_counts -> wheel_pid_controller -> set_motor_duty_cycle
   set_motor_duty_cycle -> roboclaw_for_motors -> motors
   ```

5. Switch `roboclaw_dry_run:=false` only after the duty-cycle signs and encoder signs are verified with the robot lifted.

## Split Cloud/Jetson Mode

The split path uses Zenoh instead of the HTTP/action-chunk path:

```text
/cam -> zenoh_camera_bridge -> camera/img_compressed
/asyncvla/goal -> zenoh_instruction_bridge -> robot/goal
cloud VLA -> vla/actions
Jetson action head -> vla/action_chunk
zenoh_action_chunk_bridge -> /asyncvla/action_chunk
encoder_odometry -> odom_pose2d
odom_action_chunk_tracker -> wheel_velocity_reference
wheel_pid_controller -> set_motor_duty_cycle
roboclaw_for_motors -> motors
```

On the cloud machine, run the slow VLA process with a public/listening Zenoh endpoint. In `run_vla.py`, the endpoint should be:

```python
z_conf.insert_json5("listen/endpoints", '["tcp/0.0.0.0:7447"]')
```

On the Jetson, install Zenoh and run the ROS side:

```bash
pip install eclipse-zenoh
source install/setup.bash
ros2 launch asclinic_vla asclinic_vla_split.launch.py \
  zenoh_connect_endpoint:=tcp/CLOUD_IP:7447 \
  use_goal_publisher:=true \
  roboclaw_dry_run:=true
```

Run the Jetson action head in another terminal:

```bash
source install/setup.bash
ros2 run asclinic_vla split_action_head -- \
  --connect tcp/CLOUD_IP:7447 \
  --vla-path /path/to/AsyncVLA_release \
  --dataset-config /path/to/config_nav/dataset_config.yaml
```

Set `roboclaw_dry_run:=false` only after `vla/action_chunk`, `/asyncvla/action_chunk`, `wheel_velocity_reference`, and motor direction signs are verified.
