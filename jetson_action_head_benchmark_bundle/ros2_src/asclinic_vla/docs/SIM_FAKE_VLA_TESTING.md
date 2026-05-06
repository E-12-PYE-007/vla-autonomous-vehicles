# Testing the ASClinic VLA Stack in Earthrover Simulation

This test path lets you exercise the ASClinic VLA ROS stack without running the
real AsyncVLA model. A fake model node publishes `ActionChunk` pose trajectories,
the odometry-aware tracker converts those chunks into wheel references, and a
simulation adapter converts wheel references into `/cmd_vel` for Gazebo.

## What This Tests

- Gazebo world launch from `earthrover_vla_*`.
- Simulator camera publishing on `/cam`.
- Simulator odometry publishing on `/odom`.
- Fake VLA trajectory output on `/asyncvla/action_chunk`.
- Odometry feedback into `odom_action_chunk_tracker`.
- Wheel-reference output on `wheel_velocity_reference`.
- Final simulated robot motion through `/cmd_vel`.

## Build

From the ROS 2 workspace root:

```bash
colcon build --packages-select \
  asclinic_vla_interfaces \
  asclinic_vla \
  earthrover_vla_description \
  earthrover_vla_simulation \
  earthrover_vla_bringup
source install/setup.bash
```

## Launch a Fake-VLA Simulation Test

```bash
ros2 launch asclinic_vla asclinic_vla_sim_fake.launch.py
```

Useful launch arguments:

```bash
ros2 launch asclinic_vla asclinic_vla_sim_fake.launch.py \
  worldfile:=empty_office_square.sdf \
  fake_pattern:=straight
```

Supported fake patterns:

- `straight`
- `left_arc`
- `right_arc`
- `s_curve`
- `stop`

Available world files are under:

```bash
src/earthrover_vla_simulation/worlds/templates
```

## Topic Checks

In separate terminals, source the workspace and inspect the main interfaces:

```bash
ros2 topic hz /cam
ros2 topic echo /odom --once
ros2 topic echo /odom_pose2d --once
ros2 topic echo /asyncvla/action_chunk --once
ros2 topic echo /wheel_velocity_reference
ros2 topic echo /cmd_vel
```

Expected flow:

```text
/cam and /odom come from Gazebo
/asyncvla/action_chunk comes from fake_action_chunk_publisher
/odom_pose2d comes from odom_to_pose2d
/wheel_velocity_reference comes from odom_action_chunk_tracker
/cmd_vel comes from wheel_reference_to_cmd_vel
Gazebo DiffDrive consumes /cmd_vel and moves the robot
```

## Interpreting Problems

If `/cam` or `/odom` is missing, the Gazebo bridge or Earthrover simulation
launch is the first place to check.

If `/asyncvla/action_chunk` is missing, the fake VLA publisher did not start or
the package was not rebuilt after adding the node.

If `/wheel_velocity_reference` is zero while action chunks are publishing, check
that `/odom_pose2d` is publishing. The odometry-aware tracker waits for pose
feedback before tracking a chunk.

If `/cmd_vel` is nonzero but the robot does not move, check the simulator bridge
for the `cmd_vel` ROS-to-Gazebo mapping and confirm the robot spawned.

## Switching Back to the Real Split VLA Path

The fake test path only replaces the model output. The real split path should
still publish the same downstream `ActionChunk` interface:

```text
Zenoh/cloud/action-head -> /asyncvla/action_chunk
```

Once the real model is available, stop launching `fake_action_chunk_publisher`
and use the split Zenoh nodes instead. The tracker and downstream control path
do not need to change.
