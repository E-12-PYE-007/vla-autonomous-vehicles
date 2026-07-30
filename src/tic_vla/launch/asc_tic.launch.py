#!/usr/bin/env python3
"""TIC-VLA hardware launch — inference nodes only (desktop/GPU machine).

Run this alongside `ros2 launch asc asc.launch.py` on the robot.

Usage:
    ros2 launch tic_vla asc_tic.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # Preprocesses /cam → /tic_vla/pixel_values (run on Jetson alongside asc.launch.py)
            Node(
                package="tic_vla",
                executable="image_processing",
                name="tic_vla_image_processing",
                output="screen",
            ),
            # VLM backbone — subscribes /tic_vla/pixel_values, publishes kv_cache + image_tokens
            # Prompts for a language instruction on startup.
            Node(
                package="tic_vla",
                executable="sys2",
                name="sys2",
                output="screen",
            ),
            # Action Expert — subscribes kv_cache + image_tokens + odom, publishes /ticvla/action_chunk
            Node(
                package="tic_vla",
                executable="sys1",
                name="sys1",
                output="screen",
            ),
        ]
    )
