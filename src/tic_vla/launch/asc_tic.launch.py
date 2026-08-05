#!/usr/bin/env python3
"""TIC-VLA inference nodes only — no simulator, no control stack.

Inference nodes only. Pair with a control launch running elsewhere.

On rcp the robot is reached over an SSH tunnel, which needs rmw_zenoh_cpp and a matching
ROS_DOMAIN_ID. Both are set in the shell rather than here — see the networking section of
the README. Launching this file starts no router and changes no environment.

Pair with:
    hardware:  ros2 launch asc asc.launch.py          (on the robot)
    sim:       prefer the per-stack sim launches, which include the control stack

Usage:
    ros2 launch tic_vla asc_tic.launch.py device:=rcp
    ros2 launch tic_vla asc_tic.launch.py device:=dsk
    ros2 launch tic_vla asc_tic.launch.py device:=rcp goal:="Find the red door"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Model weight locations per machine. Add a new machine by adding a row here.
DEVICE_PATHS = {
    "rcp": {
        "vlm": "/vla_storage/capstone/code/ticvla/InternVL3-1B",
        "checkpoint": "/vla_storage/capstone/code/ticvla/TIC-VLA-model.ckpt",
    },
    "dsk": {
        "vlm": "/home/vla-cap/capstone/code/ticvla/InternVL3-1B",
        "checkpoint": "/home/vla-cap/capstone/code/ticvla/TIC-VLA-model.ckpt",
    },
}


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration("device").perform(context)
    if device not in DEVICE_PATHS:
        raise RuntimeError(
            f"Unknown device '{device}'. Valid options: {', '.join(sorted(DEVICE_PATHS))}"
        )
    vlm_path = DEVICE_PATHS[device]["vlm"]
    checkpoint_path = DEVICE_PATHS[device]["checkpoint"]

    return [
        # Preprocesses /cam → /tic_vla/pixel_values
        Node(
            package="tic_vla",
            executable=f"image_processing_{device}",
            name="tic_vla_image_processing",
            output="screen",
        ),
        # VLM backbone — subscribes /tic_vla/pixel_values, publishes kv_cache + image_tokens
        Node(
            package="tic_vla",
            executable=f"sys2_{device}",
            name="sys2",
            output="screen",
            parameters=[
                {"vlm_path": vlm_path, "checkpoint_path": checkpoint_path},
                {"instruction": LaunchConfiguration("goal")},
            ],
        ),
        # Action Expert — subscribes kv_cache + image_tokens + odom,
        # publishes /ticvla/action_chunk
        Node(
            package="tic_vla",
            executable=f"sys1_{device}",
            name="sys1",
            output="screen",
            parameters=[{"checkpoint_path": checkpoint_path}],
        ),
    ]


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
                "goal",
                default_value="Go to the yellow bin.",
                description="Language instruction for TIC-VLA.",
            ),
            # Silence TensorFlow's startup noise (NUMA, cuDNN/cuFFT/cuBLAS factory,
            # TF-TRT). TF is pulled in transitively but never used for inference.
            SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3"),
            SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
