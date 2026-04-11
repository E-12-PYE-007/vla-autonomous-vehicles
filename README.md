<<<<<<< HEAD
# vla-autonomous-vehicles
E-12-PYE-007-Vision-Language-Action Models for Autonomous Vehicles
=======
# Earthrover VLA Workspace

This workspace is split into separate ROS 2 packages for robot description, simulation, and top-level bringup.

## Workspace Architecture

The `src/` folder currently contains three packages:

- `earthrover_vla_description`
  Stores the base robot model and package metadata for the physical platform.
  Package structure:
  `urdf/`
  Contains the core Xacro / URDF description of the robot, including links, joints, geometry, and inertial properties.

- `earthrover_vla_simulation`
  Stores simulation-specific assets, Gazebo configuration, and the launch logic needed to run the robot in Gazebo Sim.
  Package structure:
  `urdf/`
  Contains simulation-oriented robot description overlays, including the top-level simulation Xacro and Gazebo plugin / sensor extensions.
  `worlds/`
  Contains Gazebo world definitions used for simulation scenarios.
  `config/`
  Contains runtime configuration files, such as ROS-Gazebo bridge settings.
  `launch/`
  Contains launch files responsible for starting the simulation stack.

- `earthrover_vla_bringup`
  Provides the top-level entrypoints for launching the overall system and selecting between operating modes such as simulation and hardware.
  Package structure:
  `launch/`
  Contains wrapper launch files that route to the appropriate lower-level launch path for the chosen mode.

The dependency flow is:

`earthrover_vla_description` -> base robot model

`earthrover_vla_simulation` -> includes the description package and adds Gazebo-specific configuration

`earthrover_vla_bringup` -> includes the simulation launch file, and later can branch to hardware bringup

## Build And Source

From the workspace root:

```bash
cd ~/vla-capstone-ws
colcon build
source install/setup.bash
```

If you only want to rebuild these packages:

```bash
colcon build --packages-select \
  earthrover_vla_description \
  earthrover_vla_simulation \
  earthrover_vla_bringup
source install/setup.bash
```

After editing launch files, Xacro files, worlds, package metadata, or CMake install rules, rebuild and source again before testing.

## Clean Rebuild

If you want to force a fresh rebuild of the workspace:

```bash
rm -rf build install log
colcon build
source install/setup.bash
```

## Launching

The recommended entrypoint is the bringup package:

```bash
ros2 launch earthrover_vla_bringup launch.py
```

This defaults to simulation mode.

You can also launch the simulation package directly:

```bash
ros2 launch earthrover_vla_simulation sim.launch.py
```

## Sim Mode

To launch simulation explicitly through the bringup wrapper:

```bash
ros2 launch earthrover_vla_bringup launch.py hardware:=false
```

This will:

- expand `earthrover_vla_simulation/urdf/robot_sim.xacro`
- launch Gazebo Sim with `worlds/empty_world_cam.sdf`
- spawn the robot from `robot_description`
- start `robot_state_publisher`
- start the `ros_gz_bridge` parameter bridge

## Hardware Mode

Hardware mode is not implemented yet.

For now, this command just prints a TODO message:

```bash
ros2 launch earthrover_vla_bringup launch.py hardware:=true
```

## Notes

- The simulation Xacro uses package-based includes, so the workspace must be built and sourced before those lookups work.
- The world file currently used by simulation is `empty_world_cam.sdf`.
- The top-level bringup script is intentionally simple and acts as the mode selector for future sim and hardware launch paths.
>>>>>>> a88546b (Initial commit - basic package structure and setup for sim)
