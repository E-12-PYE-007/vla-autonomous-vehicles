#!/usr/bin/env python3
"""TIC-VLA in simulation, everything on one machine.

Brings up the simulator, the asc control nodes and the TIC-VLA inference nodes. The
controller is pinned to tic_controller here — it is a port of TIC-VLA's benchmark driver
(V_MAX 0.30, 1.0 m arc-length lookahead, bearing feedback + curvature feedforward) and is
not interchangeable with AsyncVLA's.

Run from a shell with the tic-vla conda env active, since the sys1/sys2 wrapper scripts
resolve their interpreter from PATH:

    conda activate tic-vla

    ros2 launch tic_vla sim_tic.launch.py device:=dsk
    ros2 launch tic_vla sim_tic.launch.py device:=dsk sim:=isaac
    ros2 launch tic_vla sim_tic.launch.py device:=rcp goal:="Find the red door"

sim:=gazebo starts Gazebo here. sim:=isaac expects Isaac to be running already,
publishing sensor_msgs/Image on /cam_raw and nav_msgs/Odometry on /sim_odom, and
subscribing geometry_msgs/Twist on /cmd_vel.

The controller is fixed here; change it in asc_sim.launch.py, which this includes.

For hardware, or to split sim and inference across two machines, use asc_tic.launch.py
instead.

"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    asc_sim_launch = os.path.join(
        get_package_share_directory("asc"), "launch", "asc_sim.launch.py"
    )
    tic_inference_launch = os.path.join(
        get_package_share_directory("tic_vla"), "launch", "asc_tic.launch.py"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="dsk",
                description="Which machine this is running on: dsk | rcp. Selects the "
                            "model paths and the node wrapper scripts.",
            ),
            DeclareLaunchArgument(
                "sim",
                default_value="gazebo",
                choices=["gazebo", "isaac"],
                description="Which simulator to pair with. gazebo is started here; isaac "
                            "runs separately and only needs the camera bridge.",
            ),
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin.",
                description="Language instruction for TIC-VLA.",
            ),
            # --- Simulator + control ---
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(asc_sim_launch),
                launch_arguments={
                    "sim": LaunchConfiguration("sim"),
                    "controller": "tic_controller",
                }.items(),
            ),
            # --- Inference ---
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(tic_inference_launch),
                launch_arguments={
                    "device": LaunchConfiguration("device"),
                    "goal": LaunchConfiguration("goal"),
                }.items(),
            ),
        ]
    )
