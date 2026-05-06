#!/usr/bin/env python3
"""Launch the Earthrover simulator with fake VLA outputs.

This launch file tests the ASClinic VLA control path without the real model:

fake ActionChunk -> odom action-chunk tracker -> wheel refs -> /cmd_vel -> Gazebo

The camera and odometry still come from the simulator, so it is useful for
checking that perception inputs, reference trajectory handling, and simulated
motion are all connected correctly.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_asclinic_vla = get_package_share_directory('asclinic_vla')
    pkg_earthrover_bringup = get_package_share_directory('earthrover_vla_bringup')

    default_config = os.path.join(
        pkg_asclinic_vla,
        'config',
        'asclinic_vla_sim_fake.yaml',
    )
    earthrover_launch = os.path.join(
        pkg_earthrover_bringup,
        'launch',
        'launch.py',
    )

    config_file = LaunchConfiguration('config_file')
    worldfile = LaunchConfiguration('worldfile')
    fake_pattern = LaunchConfiguration('fake_pattern')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='YAML config for the fake VLA simulation stack.',
        ),
        DeclareLaunchArgument(
            'worldfile',
            default_value='empty_world_cam.sdf',
            description='Earthrover world file from earthrover_vla_simulation/worlds/templates.',
        ),
        DeclareLaunchArgument(
            'fake_pattern',
            default_value='straight',
            description='Fake VLA path: straight, left_arc, right_arc, s_curve, or stop.',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(earthrover_launch),
            launch_arguments={
                'mode': 'sim',
                'worldfile': worldfile,
            }.items(),
        ),
        Node(
            package='asclinic_vla',
            executable='fake_action_chunk_publisher',
            name='fake_action_chunk_publisher',
            output='screen',
            parameters=[config_file, {'pattern': fake_pattern}],
        ),
        Node(
            package='asclinic_vla',
            executable='odom_to_pose2d',
            name='odom_to_pose2d',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='asclinic_vla',
            executable='odom_action_chunk_tracker',
            name='odom_action_chunk_tracker',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='asclinic_vla',
            executable='wheel_reference_to_cmd_vel',
            name='wheel_reference_to_cmd_vel',
            output='screen',
            parameters=[config_file],
        ),
    ])
