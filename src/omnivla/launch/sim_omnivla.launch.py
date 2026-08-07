#!/usr/bin/env python3
"""OmniVLA in simulation, everything on one machine.

Brings up the simulator, the asc control nodes and the OmniVLA inference node.
The controller is pinned to outer_loop_controller — tic_controller's 1.0 m lookahead
oversteers badly on shorter action chunks, and having picked the wrong one by default
has cost real debugging time before.

Run from a shell with the omnivla conda env active, since the node wrapper script
resolves its interpreter from PATH:

    conda activate omnivla

    ros2 launch omnivla sim_omnivla.launch.py device:=dsk
    ros2 launch omnivla sim_omnivla.launch.py device:=dsk sim:=isaac
    ros2 launch omnivla sim_omnivla.launch.py device:=rcp goal:="Find the red door"

sim:=gazebo starts Gazebo here. sim:=isaac expects Isaac to be running already,
publishing sensor_msgs/Image on /cam_raw and nav_msgs/Odometry on /sim_odom, and
subscribing geometry_msgs/Twist on /cmd_vel.

For hardware, or to split sim and inference across two machines, run
asc.launch.py on the robot and omnivla.launch.py on the inference machine separately.
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
    omnivla_launch = os.path.join(
        get_package_share_directory("omnivla"), "launch", "omnivla.launch.py"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="dsk",
                description="Which machine this is running on: dsk | rcp. "
                            "Selects the model path and wrapper script.",
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
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
            ),
            # --- Simulator + control ---
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(asc_sim_launch),
                launch_arguments={
                    "sim": LaunchConfiguration("sim"),
                    "controller": "outer_loop_controller",
                }.items(),
            ),
            # --- Inference ---
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(omnivla_launch),
                launch_arguments={
                    "device": LaunchConfiguration("device"),
                    "goal": LaunchConfiguration("goal"),
                }.items(),
            ),
        ]
    )
