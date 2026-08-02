#!/usr/bin/env python3
"""TIC-VLA hardware launch — inference nodes only (rcp: this remote desktop).

Run this alongside `ros2 launch asc asc.launch.py` on the robot.

Usage:
    ros2 launch tic_vla asc_tic_rcp.launch.py
"""

import shutil
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

VLM_PATH = "/vla_storage/capstone/code/ticvla/InternVL3-1B"
CHECKPOINT_PATH = "/vla_storage/capstone/code/ticvla/TIC-VLA-model.ckpt"

# rmw_zenoh_cpp needs a router process before any node can init. Skip starting our
# own if one is already up, otherwise zenoh fails to bind its port.
ROUTER_RUNNING = subprocess.run(["pgrep", "-x", "rmw_zenohd"], capture_output=True).returncode == 0


def zenoh_router_actions(nodes):
    """Wrap nodes so a zenoh router is up first and torn down on shutdown."""
    if ROUTER_RUNNING or shutil.which("ros2") is None:
        return nodes
    router = ExecuteProcess(
        cmd=["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
        name="rmw_zenohd",
        output="screen",
        # Launch SIGINTs the router on Ctrl-C; escalate if it doesn't exit promptly.
        sigterm_timeout="5",
        sigkill_timeout="10",
    )
    # Give the router a moment to bind before nodes try to connect.
    return [router, TimerAction(period=2.0, actions=nodes)]


def generate_launch_description():
    nodes = [
        # Preprocesses /cam → /tic_vla/pixel_values (run on Jetson alongside asc.launch.py)
        Node(
            package="tic_vla",
            executable="image_processing_rcp",
            name="tic_vla_image_processing",
            output="screen",
        ),
        # VLM backbone — subscribes /tic_vla/pixel_values, publishes kv_cache + image_tokens
        Node(
            package="tic_vla",
            executable="sys2_rcp",
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
            executable="sys1_rcp",
            name="sys1",
            output="screen",
            parameters=[{"checkpoint_path": CHECKPOINT_PATH}],
        ),
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "domain_id",
                default_value="1",
                description="ROS_DOMAIN_ID — must match the robot.",
            ),
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin.",
                description="Language instruction for TIC-VLA.",
            ),
            # Set before anything spawns so nodes and the router inherit them.
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_zenoh_cpp"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", LaunchConfiguration("domain_id")),
        ]
        + zenoh_router_actions(nodes)
    )
