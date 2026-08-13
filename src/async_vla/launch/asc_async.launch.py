#!/usr/bin/env python3
"""AsyncVLA inference nodes only — no simulator, no control stack.

Inference nodes only. Pair with a control launch running elsewhere.

On rcp the robot is reached over an SSH tunnel, which needs rmw_zenoh_cpp and a matching
ROS_DOMAIN_ID. Both are set in the shell rather than here — see the networking section of
the README. Launching this file starts no router and changes no environment.

Pair with:
    hardware:  ros2 launch asc asc.launch.py          (on the robot)
    sim:       prefer the per-stack sim launches, which include the control stack

Usage:
    ros2 launch async_vla asc_async.launch.py device:=rcp
    ros2 launch async_vla asc_async.launch.py device:=dsk
    ros2 launch async_vla asc_async.launch.py device:=rcp goal:="Find the red door"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Model weight locations per machine. Add a new machine by adding a row here.
DEVICE_PATHS = {
    "rcp": "/vla_storage/capstone/code/asyncvla/AsyncVLA/AsyncVLA_release",
    "dsk": "/home/vla-cap/AsyncVLA/AsyncVLA_release",
}


def launch_setup(context, *args, **kwargs):
    device = LaunchConfiguration("device").perform(context)
    if device not in DEVICE_PATHS:
        raise RuntimeError(
            f"Unknown device '{device}'. Valid options: {', '.join(sorted(DEVICE_PATHS))}"
        )
    vla_path = DEVICE_PATHS[device]

    return [
        # Backbone VLA — subscribes /cam, publishes /asyncvla/hidden_state
        Node(
            package="async_vla",
            executable=f"sys2_{device}",
            name="sys2",
            output="screen",
            parameters=[
                {"vla_path": vla_path},
                {"goal": LaunchConfiguration("goal")},
                {"sim": LaunchConfiguration("sim")},
            ],
        ),
        # Edge adapter — subscribes /asyncvla/hidden_state + /cam,
        # publishes /asyncvla/action_chunk at 3 Hz
        Node(
            package="async_vla",
            executable=f"sys1_{device}",
            name="sys1",
            output="screen",
            parameters=[
                {"shead_path": vla_path},
                {"sim": LaunchConfiguration("sim")},
            ],
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
                default_value="Go to the yellow bin",
                description="Language goal for the VLA.",
            ),
            DeclareLaunchArgument(
                "sim",
                default_value="",
                description="Simulator in use. 'isaac' makes sys2 subscribe to /vla/cam "
                            "(sensor_msgs/Image); otherwise it uses /cam (ImageWithSeqNum).",
            ),
            # Silence TensorFlow's startup noise (NUMA, cuDNN/cuFFT/cuBLAS factory,
            # TF-TRT). TF is pulled in transitively but never used for inference.
            SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3"),
            SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "0"),
            OpaqueFunction(function=launch_setup),
        ]
    )
