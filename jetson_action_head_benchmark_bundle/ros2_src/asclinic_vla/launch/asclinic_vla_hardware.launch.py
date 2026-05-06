from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration('config_file')
    use_vla_http_client = LaunchConfiguration('use_vla_http_client')
    use_goal_publisher = LaunchConfiguration('use_goal_publisher')
    goal_mode = LaunchConfiguration('goal_mode')
    goal_text = LaunchConfiguration('goal_text')
    goal_image_path = LaunchConfiguration('goal_image_path')
    use_odom_tracker = LaunchConfiguration('use_odom_tracker')
    roboclaw_dry_run = LaunchConfiguration('roboclaw_dry_run')

    default_config = PathJoinSubstitution([
        FindPackageShare('asclinic_vla'),
        'config',
        'asclinic_vla_hardware.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='YAML config for ASClinic VLA hardware nodes.',
        ),
        DeclareLaunchArgument(
            'use_vla_http_client',
            default_value='true',
            description='Start the robot-side HTTP client for cloud VLA inference.',
        ),
        DeclareLaunchArgument(
            'use_goal_publisher',
            default_value='false',
            description='Start the ASClinic VLA goal publisher.',
        ),
        DeclareLaunchArgument(
            'goal_mode',
            default_value='text',
            description='Goal publisher mode: text or image.',
        ),
        DeclareLaunchArgument(
            'goal_text',
            default_value='',
            description='Text instruction to publish when goal_mode:=text.',
        ),
        DeclareLaunchArgument(
            'goal_image_path',
            default_value='',
            description='Image path on the robot to publish when goal_mode:=image.',
        ),
        DeclareLaunchArgument(
            'roboclaw_dry_run',
            default_value='false',
            description='Run Roboclaw node without opening hardware.',
        ),
        DeclareLaunchArgument(
            'use_odom_tracker',
            default_value='true',
            description='Use encoder odometry feedback for ActionChunk trajectory tracking.',
        ),
        Node(
            package='asclinic_vla',
            executable='goal_publisher',
            name='goal_publisher',
            output='screen',
            parameters=[
                config_file,
                {
                    'mode': goal_mode,
                    'goal_text': goal_text,
                    'image_path': goal_image_path,
                },
            ],
            condition=IfCondition(use_goal_publisher),
        ),
        Node(
            package='asclinic_vla',
            executable='asclinic_camera_capture',
            name='asclinic_camera_capture',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='asclinic_vla',
            executable='vla_http_client',
            name='vla_http_client',
            output='screen',
            parameters=[config_file],
            condition=IfCondition(use_vla_http_client),
        ),
        Node(
            package='asclinic_vla',
            executable='action_chunk_pure_pursuit',
            name='action_chunk_pure_pursuit',
            output='screen',
            parameters=[config_file],
            condition=UnlessCondition(use_odom_tracker),
        ),
        Node(
            package='asclinic_vla',
            executable='encoder_odometry',
            name='encoder_odometry',
            output='screen',
            parameters=[config_file],
            condition=IfCondition(use_odom_tracker),
        ),
        Node(
            package='asclinic_vla',
            executable='odom_action_chunk_tracker',
            name='odom_action_chunk_tracker',
            output='screen',
            parameters=[config_file],
            condition=IfCondition(use_odom_tracker),
        ),
        Node(
            package='asclinic_vla',
            executable='wheel_pid_controller',
            name='wheel_pid_controller',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='asclinic_vla',
            executable='roboclaw_for_motors',
            name='roboclaw_for_motors',
            output='screen',
            parameters=[
                config_file,
                {'dry_run': roboclaw_dry_run},
            ],
        ),
    ])
