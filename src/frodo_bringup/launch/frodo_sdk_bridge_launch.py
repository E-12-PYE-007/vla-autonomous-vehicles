from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='frodo_bringup',
            executable='frodo_sdk_bridge',
            name='frodo_sdk_bridge',
            output='screen',
            parameters=[{
                'base_url': 'http://localhost:8000',
                'control_rate_hz': 10.0,
                'telemetry_rate_hz': 5.0,
                'camera_rate_hz': 2.0,
                'max_linear_mps': 1.111,
                'max_angular_cmd': 1.0,
                'lamp': 0,
            }],
        )
    ])
