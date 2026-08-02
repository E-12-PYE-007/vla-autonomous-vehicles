#!/usr/bin/env python3
"""TIC-VLA hardware launch — inference nodes only (dsk: lab desktop, vla-cap).

Run this alongside `ros2 launch asc asc.launch.py` on the robot.

Usage:
    ros2 launch tic_vla asc_tic_dsk.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VLM_PATH = "/home/vla-cap/capstone/code/ticvla/InternVL3-1B"
CHECKPOINT_PATH = "/home/vla-cap/capstone/code/ticvla/TIC-VLA-model.ckpt"


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin.",
                description="Language instruction for TIC-VLA.",
            ),
            # Preprocesses /cam → /tic_vla/pixel_values (run on Jetson alongside asc.launch.py)
            Node(
                package="tic_vla",
                executable="image_processing_dsk",
                name="tic_vla_image_processing",
                output="screen",
            ),
            # VLM backbone — subscribes /tic_vla/pixel_values, publishes kv_cache + image_tokens
            Node(
                package="tic_vla",
                executable="sys2_dsk",
                name="sys2",
                output="screen",
                parameters=[
                    {"vlm_path": VLM_PATH, "checkpoint_path": CHECKPOINT_PATH},
                    {"instruction": LaunchConfiguration("goal")},
                ],
            ),
            # Action Expert — subscribes kv_cache + image_tokens + odom, publishes /ticvla/action_chunk
            Node(
                package="tic_vla",
                executable="sys1_dsk",
                name="sys1",
                output="screen",
                parameters=[{"checkpoint_path": CHECKPOINT_PATH}],
            ),
        ]
    )
