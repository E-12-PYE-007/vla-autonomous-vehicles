#!/usr/bin/env python3

"""AsyncVLA and sim launch (to be ran on remote desktop)

Starts Gazebo (via earthrover_vla_bringup), the asc control nodes in
sim-compatible configuration, and the AsyncVLA inference nodes.

Usage:
    ros2 launch async_vla sim.launch.py
    ros2 launch async_vla sim.launch.py worldfile:=unempty_office_square.sdf
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
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
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
            # Subscribes to /odom (Odometry from Gazebo bridge) and republishes as odom_pose2d (Pose2D)
            Node(
                package="asc",
                executable="odometry",
                name="odometry",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
            # PD waypoint controller — reference AsyncVLA controller, publishes /cmd_vel at 10 Hz
            Node(
                package="asc",
                executable="async_pd_controller",
                name="async_pd_controller",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
            # --- AsyncVLA inference nodes ---
            Node(
                package="async_vla",
                executable="sys2",
                name="sys2",
                output="screen",
                parameters=[{"goal": LaunchConfiguration("goal")}, {"use_sim": True}],
            ),
            Node(
                package="async_vla",
                executable="sys1",
                name="sys1",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
            # Snapshots sys1 + sys2 action chunks to CSV once per second for offline plotting.
            Node(
                package="async_vla",
                executable="store_action_chunks",
                name="store_action_chunks",
                output="screen",
            ),
        ]
    )