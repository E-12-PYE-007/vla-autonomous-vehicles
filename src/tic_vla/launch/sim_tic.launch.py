#!/usr/bin/env python3
"""TIC-VLA sim launch — Gazebo + control nodes + inference nodes (single machine).

Usage:
    ros2 launch tic_vla sim_tic.launch.py
    ros2 launch tic_vla sim_tic.launch.py worldfile:=unempty_office_square.sdf
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VLM_PATH = "/vla_storage/capstone/code/ticvla/InternVL3-1B"
CHECKPOINT_PATH = "/vla_storage/capstone/code/ticvla/TIC-VLA-model.ckpt"


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
                default_value="Go to the yellow bin.",
                description="Language instruction for TIC-VLA.",
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
                executable="outer_loop_controller",
                name="odom_action_chunk_tracker",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
            # --- TIC-VLA inference nodes ---
            Node(
                package="tic_vla",
                executable="image_processing_rcp",
                name="tic_vla_image_processing",
                output="screen",
                parameters=[{"use_sim": True}],
            ),
            Node(
                package="tic_vla",
                executable="sys2_rcp",
                name="sys2",
                output="screen",
                parameters=[
                    {"instruction": LaunchConfiguration("goal")},
                    {"use_sim": True},
                    {"vlm_path": VLM_PATH, "checkpoint_path": CHECKPOINT_PATH},
                ],
            ),
            Node(
                package="tic_vla",
                executable="sys1_rcp",
                name="sys1",
                output="screen",
                parameters=[{"use_sim": True}, {"checkpoint_path": CHECKPOINT_PATH}],
            ),
        ]
    )
