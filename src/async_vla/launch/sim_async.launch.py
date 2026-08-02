#!/usr/bin/env python3

"""AsyncVLA + simulator, all on one machine.

Starts Gazebo (via earthrover_vla_bringup), the asc control nodes in
sim-compatible configuration, and the AsyncVLA inference nodes.

For a split setup (sim on one machine, inference on another) use
asc_async.launch.py with use_sim:=true instead, alongside `ros2 launch asc asc_sim.launch.py`.

Usage:
    ros2 launch async_vla sim_async.launch.py
    ros2 launch async_vla sim_async.launch.py device:=dsk
    ros2 launch async_vla sim_async.launch.py worldfile:=unempty_office_square.sdf
    ros2 launch async_vla sim_async.launch.py goal:="Find the red door"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Model weight locations per machine. Keep in sync with asc_async.launch.py.
DEVICE_PATHS = {
    "rcp": "/vla_storage/capstone/code/asyncvla/AsyncVLA/AsyncVLA_release",
    "dsk": "/home/vla-cap/AsyncVLA/AsyncVLA_release",
}


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration("device").perform(context)
    if device not in DEVICE_PATHS:
        raise RuntimeError(
            f"Unknown device '{device}'. Valid options: {', '.join(sorted(DEVICE_PATHS))}"
        )
    vla_path = DEVICE_PATHS[device]

    earthrover_bringup_launch = os.path.join(
        get_package_share_directory("earthrover_vla_bringup"),
        "launch",
        "launch.py",
    )

    return [
        # --- Simulator ---
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(earthrover_bringup_launch),
            launch_arguments={
                "mode": "sim",
                "worldfile": LaunchConfiguration("worldfile"),
            }.items(),
        ),
        # --- ASC control nodes (sim mode) ---
        # Subscribes to /odom (Odometry from Gazebo bridge), republishes as odom_pose2d
        Node(
            package="asc",
            executable="odometry",
            name="odometry",
            output="screen",
            parameters=[{"use_sim": True}],
        ),
        # Converts ActionChunk + odom_pose2d into /cmd_vel (Twist) for Gazebo
        Node(
            package="asc",
            executable="outer_loop_controller",
            name="outer_loop_controller",
            output="screen",
            parameters=[{"use_sim": True}],
        ),
        # --- AsyncVLA inference nodes ---
        Node(
            package="async_vla",
            executable=f"sys2_{device}",
            name="sys2",
            output="screen",
            parameters=[
                {"goal": LaunchConfiguration("goal")},
                {"vla_path": vla_path},
            ],
        ),
        Node(
            package="async_vla",
            executable=f"sys1_{device}",
            name="sys1",
            output="screen",
            parameters=[{"shead_path": vla_path}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="rcp",
                description="Which machine this is running on: "
                            f"{' | '.join(sorted(DEVICE_PATHS))}. "
                            "Selects the model paths and the node wrapper scripts.",
            ),
            DeclareLaunchArgument(
                "worldfile",
                default_value="unempty_office_square.sdf",
                description="Gazebo world file (relative to earthrover_vla_simulation/worlds/templates).",
            ),
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
            ),
            # Silence TensorFlow's startup noise (NUMA, cuDNN/cuFFT/cuBLAS factory,
            # TF-TRT). TF is pulled in transitively but never used for inference.
            SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3"),
            SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
