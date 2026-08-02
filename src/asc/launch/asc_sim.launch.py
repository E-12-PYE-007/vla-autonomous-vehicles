#!/usr/bin/env python3
"""ASC sim launch — Gazebo + control nodes only (no inference).

Run this on the VM alongside inference nodes on the desktop:
    Desktop: ros2 launch async_vla asc_async.launch.py
             ros2 launch tic_vla asc_tic.launch.py

Usage:
    ros2 launch asc sim.launch.py
    ros2 launch asc sim.launch.py worldfile:=unempty_office_square.sdf
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    worldfile = LaunchConfiguration("worldfile")

    earthrover_bringup_launch = os.path.join(
        get_package_share_directory("earthrover_vla_bringup"),
        "launch",
        "launch.py",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "worldfile",
                default_value="unempty_office_square.sdf",
                description="Gazebo world file (relative to earthrover_vla_simulation/worlds/templates).",
            ),
            # --- Simulator ---
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(earthrover_bringup_launch),
                launch_arguments={
                    "mode": "sim",
                    "worldfile": worldfile,
                }.items(),
            ),
            # --- ASC control nodes (sim mode) ---
            Node(
                package="asc",
                executable="odometry",
                name="encoder_odometry",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
            Node(
                package="asc",
                executable="pure_pursuit_controller",
                name="pure_pursuit_controller",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
        ]
    )
