from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # This launch file starts the ROS side of the split architecture on the
    # Jetson. The action-head model runner is intentionally a separate process
    # because it has heavier Python/CUDA dependencies and clearer logging needs.
    config_file = LaunchConfiguration('config_file')
    use_goal_publisher = LaunchConfiguration('use_goal_publisher')
    goal_mode = LaunchConfiguration('goal_mode')
    goal_text = LaunchConfiguration('goal_text')
    goal_image_path = LaunchConfiguration('goal_image_path')
    roboclaw_dry_run = LaunchConfiguration('roboclaw_dry_run')
    zenoh_connect_endpoint = LaunchConfiguration('zenoh_connect_endpoint')

    default_config = PathJoinSubstitution([
        FindPackageShare('asclinic_vla'),
        'config',
        'asclinic_vla_split.yaml',
    ])

    return LaunchDescription([
        # Runtime knobs. In normal Jetson use, the main one to set is
        # zenoh_connect_endpoint:=tcp/CLOUD_IP:7447.
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='YAML config for split cloud/Jetson ASClinic VLA nodes.',
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
            description='Image path on the Jetson to publish when goal_mode:=image.',
        ),
        DeclareLaunchArgument(
            'roboclaw_dry_run',
            default_value='false',
            description='Run Roboclaw node without opening hardware.',
        ),
        DeclareLaunchArgument(
            'zenoh_connect_endpoint',
            default_value='tcp/127.0.0.1:7447',
            description='Zenoh endpoint for the cloud VLA process or local router.',
        ),
        Node(
            # Optional helper for publishing a text/image goal into ROS.
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
            # Physical camera capture on the robot. Publishes /cam.
            package='asclinic_vla',
            executable='asclinic_camera_capture',
            name='asclinic_camera_capture',
            output='screen',
            parameters=[
                config_file,
                {'zenoh_connect_endpoints': [zenoh_connect_endpoint]},
            ],
        ),
        Node(
            # Converts ROS /cam into the JSON/JPEG Zenoh payload expected by the
            # cloud VLA and the Jetson action head.
            package='asclinic_vla',
            executable='zenoh_camera_bridge',
            name='zenoh_camera_bridge',
            output='screen',
            parameters=[
                config_file,
                {'zenoh_connect_endpoints': [zenoh_connect_endpoint]},
            ],
        ),
        Node(
            # Sends ROS text/image GoalSpec messages to the single cloud goal key.
            package='asclinic_vla',
            executable='zenoh_instruction_bridge',
            name='zenoh_instruction_bridge',
            output='screen',
            parameters=[
                config_file,
                {'zenoh_connect_endpoints': [zenoh_connect_endpoint]},
            ],
        ),
        Node(
            # Receives action-head pose chunks and republishes /asyncvla/action_chunk.
            package='asclinic_vla',
            executable='zenoh_action_chunk_bridge',
            name='zenoh_action_chunk_bridge',
            output='screen',
            parameters=[
                config_file,
                {'zenoh_connect_endpoints': [zenoh_connect_endpoint]},
            ],
        ),
        Node(
            # Integrates encoder deltas into odom_pose2d for feedback tracking.
            package='asclinic_vla',
            executable='encoder_odometry',
            name='encoder_odometry',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            # Converts VLA pose chunks + odometry error into wheel velocity refs.
            package='asclinic_vla',
            executable='odom_action_chunk_tracker',
            name='odom_action_chunk_tracker',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            # Closed-loop low-level wheel velocity control.
            package='asclinic_vla',
            executable='wheel_pid_controller',
            name='wheel_pid_controller',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            # Hardware driver. With roboclaw_dry_run true it does not open motors.
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
