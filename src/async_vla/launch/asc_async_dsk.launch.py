#!/usr/bin/env python3
"""AsyncVLA hardware launch — inference nodes only (dsk: lab desktop, vla-cap).

Run this alongside `ros2 launch asc asc.launch.py` on the robot.

Usage:
    ros2 launch async_vla asc_async_dsk.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VLA_PATH = "/home/vla-cap/AsyncVLA/AsyncVLA_release"


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
            ),
            # Backbone VLA — subscribes /cam, publishes /asyncvla/hidden_state
            Node(
                package="async_vla",
                executable="sys2_dsk",
                name="sys2",
                output="screen",
                parameters=[
                    {"vla_path": VLA_PATH},
                    {"goal": LaunchConfiguration("goal")},
                ],
            ),
            # Edge adapter — subscribes /asyncvla/hidden_state + /cam, publishes /asyncvla/action_chunk at 3 Hz
            Node(
                package="async_vla",
                executable="sys1_dsk",
                name="sys1",
                output="screen",
                parameters=[{"shead_path": VLA_PATH}],
            ),
        ]
    )
