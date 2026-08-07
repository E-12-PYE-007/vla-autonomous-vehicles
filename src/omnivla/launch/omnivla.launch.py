#!/usr/bin/env python3
"""OmniVLA inference node.

Runs the OmniVLA VLA inference node with a language goal. Subscribes to /cam,
publishes action chunks on /asyncvla/action_chunk.

Usage:
    ros2 launch omnivla omnivla.launch.py device:=dsk goal:="Go to the yellow bin"
    ros2 launch omnivla omnivla.launch.py device:=rcp goal:="Find the red door"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

DEVICE_PATHS = {
    "dsk": "/home/vla-cap/omniVLA-release/omnivla-finetuned-cast",
    "rcp": "/vla_storage/capstone/code/omnivla/omniVLA-release/omnivla-finetuned-cast",
}


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration("device").perform(context)
    if device not in DEVICE_PATHS:
        raise RuntimeError(
            f"Unknown device '{device}'. Valid options: {', '.join(sorted(DEVICE_PATHS))}"
        )
    vla_path = DEVICE_PATHS[device]

    return [
        Node(
            package="omnivla",
            executable=f"omnivla_{device}",
            name="omnivla",
            output="screen",
            parameters=[
                {"vla_path": vla_path},
                {"goal": LaunchConfiguration("goal")},
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="dsk",
                description=f"Which machine this is running on: "
                            f"{' | '.join(sorted(DEVICE_PATHS))}. "
                            "Selects the model path and wrapper script.",
            ),
            DeclareLaunchArgument(
                "goal",
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
            ),
            SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3"),
            SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
