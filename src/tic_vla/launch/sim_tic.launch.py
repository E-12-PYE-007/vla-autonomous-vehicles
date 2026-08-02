#!/usr/bin/env python3
"""TIC-VLA + simulator, all on one machine.

Starts Gazebo (via earthrover_vla_bringup), the asc control nodes in
sim-compatible configuration, and the TIC-VLA inference nodes.

For a split setup (sim on one machine, inference on another) use
asc_tic.launch.py with use_sim:=true instead, alongside `ros2 launch asc asc_sim.launch.py`.

Usage:
    ros2 launch tic_vla sim_tic.launch.py
    ros2 launch tic_vla sim_tic.launch.py device:=dsk
    ros2 launch tic_vla sim_tic.launch.py worldfile:=unempty_office_square.sdf
    ros2 launch tic_vla sim_tic.launch.py goal:="Find the red door"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Model weight locations per machine. Keep in sync with asc_tic.launch.py.
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

    earthrover_bringup_launch = os.path.join(
        get_package_share_directory("earthrover_vla_bringup"),
        "launch",
        "launch.py",
    )

    return [
        # --- Simulator ---
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(earthrover_bringup_launch),
            launch_arguments={
                "mode": "sim",
                "worldfile": LaunchConfiguration("worldfile"),
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
        # TIC-VLA benchmark controller (arc-length pure pursuit, port of upstream's
        # nova_carter driver). Swap executable back to outer_loop_controller for the
        # odometry-anchored tracker.
        Node(
            package="asc",
            executable="tic_controller",
            name="tic_controller",
            output="screen",
            parameters=[{"use_sim": True}],
        ),
        # --- TIC-VLA inference nodes ---
        Node(
            package="tic_vla",
            executable=f"image_processing_{device}",
            name="tic_vla_image_processing",
            output="screen",
            parameters=[{"use_sim": True}],
        ),
        Node(
            package="tic_vla",
            executable=f"sys2_{device}",
            name="sys2",
            output="screen",
            parameters=[
                {"instruction": LaunchConfiguration("goal")},
                {"use_sim": True},
                {"vlm_path": vlm_path, "checkpoint_path": checkpoint_path},
            ],
        ),
        Node(
            package="tic_vla",
            executable=f"sys1_{device}",
            name="sys1",
            output="screen",
            parameters=[{"use_sim": True}, {"checkpoint_path": checkpoint_path}],
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
                "worldfile",
                # 30.3 x 4.3 m corridor with targets 14-16 m away. The 5.3 m square room
                # is smaller than a single one of TIC-VLA's planning horizons (a 3 s plan
                # at its trained ~1.5 m/s covers ~4.5 m; it reasons out to 9 s).
                default_value="unempty_office_hallway.sdf",
                description="Gazebo world file (relative to earthrover_vla_simulation/worlds/templates).",
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
