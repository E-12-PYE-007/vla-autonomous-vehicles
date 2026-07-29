#!/usr/bin/env python3
"""AsyncVLA hardware launch — inference nodes only (desktop/GPU machine).

Run this alongside `ros2 launch asc asc.launch.py` on the robot.

Usage:
    ros2 launch async_vla asc_async.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # Backbone VLA — subscribes /cam, publishes /asyncvla/hidden_state
            # Prompts for a language goal on startup.
            Node(
                package="async_vla",
                executable="sys2",
                name="sys2",
                output="screen",
            ),
            # Edge adapter — subscribes /asyncvla/hidden_state + /cam, publishes /asyncvla/action_chunk at 3 Hz
            Node(
                package="async_vla",
                executable="sys1",
                name="sys1",
                output="screen",
            ),
        ]
    )
