#!/usr/bin/env python3
"""AsyncVLA inference nodes only — no simulator, no control stack.

Pair with a control launch running elsewhere:
    hardware:  ros2 launch asc asc.launch.py          (on the robot)
    sim:       ros2 launch asc asc_sim.launch.py      (+ pass use_sim:=true here)

Usage:
    ros2 launch async_vla asc_async.launch.py device:=rcp
    ros2 launch async_vla asc_async.launch.py device:=dsk
    ros2 launch async_vla asc_async.launch.py device:=rcp use_sim:=true
    ros2 launch async_vla asc_async.launch.py device:=rcp goal:="Find the red door"
"""

import shutil
import subprocess

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Model weight locations per machine. Add a new machine by adding a row here.
DEVICE_PATHS = {
    "rcp": "/vla_storage/capstone/code/asyncvla/AsyncVLA/AsyncVLA_release",
    "dsk": "/home/vla-cap/AsyncVLA/AsyncVLA_release",
}

# Machines that reach the robot over an SSH tunnel and therefore need a local
# zenoh router started for them.
NEEDS_ZENOH_ROUTER = {"rcp"}


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration("device").perform(context)
    if device not in DEVICE_PATHS:
        raise RuntimeError(
            f"Unknown device '{device}'. Valid options: {', '.join(sorted(DEVICE_PATHS))}"
        )
    vla_path = DEVICE_PATHS[device]

    # "true"/"false" from the command line -> real bool for the node parameter.
    use_sim = LaunchConfiguration("use_sim").perform(context).lower() in ("true", "1", "yes")

    nodes = [
        # Backbone VLA — subscribes /cam, publishes /asyncvla/hidden_state
        Node(
            package="async_vla",
            executable=f"sys2_{device}",
            name="sys2",
            output="screen",
            parameters=[
                {"vla_path": vla_path},
                {"goal": LaunchConfiguration("goal")},
                {"use_sim": use_sim},
            ],
        ),
        # Edge adapter — subscribes /asyncvla/hidden_state + /cam,
        # publishes /asyncvla/action_chunk at 3 Hz
        Node(
            package="async_vla",
            executable=f"sys1_{device}",
            name="sys1",
            output="screen",
            parameters=[{"shead_path": vla_path}, {"use_sim": use_sim}],
        ),
    ]

    if device not in NEEDS_ZENOH_ROUTER:
        return nodes

    actions = [
        # rmw_zenoh_cpp needs a router before any node can init.
        SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_zenoh_cpp"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", LaunchConfiguration("domain_id")),
    ]

    # Skip starting our own router if one is already up, otherwise zenoh fails to bind.
    router_running = subprocess.run(["pgrep", "-x", "rmw_zenohd"], capture_output=True).returncode == 0
    if router_running or shutil.which("ros2") is None:
        return actions + nodes

    router = ExecuteProcess(
        cmd=["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
        name="rmw_zenohd",
        output="screen",
        # Launch SIGINTs the router on Ctrl-C; escalate if it doesn't exit promptly.
        sigterm_timeout="5",
        sigkill_timeout="10",
    )
    # Give the router a moment to bind before nodes try to connect.
    return actions + [router, TimerAction(period=2.0, actions=nodes)]


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
                "use_sim",
                default_value="false",
                description="true when pairing with asc_sim.launch.py — Gazebo publishes "
                            "sensor_msgs/Image on /cam instead of ImageWithSeqNum.",
            ),
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
            ),
            DeclareLaunchArgument(
                "domain_id",
                default_value="1",
                description="ROS_DOMAIN_ID — must match the robot. Only applied on "
                            "devices that start their own zenoh router.",
            ),
            # Silence TensorFlow's startup noise (NUMA, cuDNN/cuFFT/cuBLAS factory,
            # TF-TRT). TF is pulled in transitively but never used for inference.
            SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3"),
            SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
