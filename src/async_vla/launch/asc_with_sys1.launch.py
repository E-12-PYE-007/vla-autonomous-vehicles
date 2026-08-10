#!/usr/bin/env python3
"""ASC hardware stack + AsyncVLA sys1 — robot side of a split deployment.

Brings up the full hardware control stack (camera, odometry, outer/inner loop
controllers, Roboclaw) and the sys1 edge adapter. sys1 subscribes to
/asyncvla/hidden_state from the workstation and publishes /asyncvla/action_chunk
locally for the outer loop controller to consume.

Both machines must be on the same network with a matching ROS_DOMAIN_ID.

    ros2 launch async_vla asc_with_sys1.launch.py

Pair with (on the workstation):
    ros2 launch async_vla sys2_only.launch.py goal:="..."
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

SHEAD_PATH = "/home/asc/asyncvla-test/AsyncVLA/AsyncVLA_release"


def generate_launch_description():
    asc_launch = os.path.join(
        get_package_share_directory("asc"), "launch", "asc.launch.py"
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(asc_launch),
        ),
        Node(
            package="async_vla",
            executable="sys1_rcp",
            name="sys1",
            output="screen",
            parameters=[{"shead_path": SHEAD_PATH}],
        ),
    ])
