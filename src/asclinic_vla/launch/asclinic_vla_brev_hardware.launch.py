from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch the Jetson ROS/hardware side for Brev + action-head-over-Zenoh."""
    config_file = LaunchConfiguration('config_file')
    use_goal_publisher = LaunchConfiguration('use_goal_publisher')
    goal_mode = LaunchConfiguration('goal_mode')
    goal_text = LaunchConfiguration('goal_text')
    goal_image_path = LaunchConfiguration('goal_image_path')
    zenoh_connect_endpoint = LaunchConfiguration('zenoh_connect_endpoint')
    roboclaw_dry_run = LaunchConfiguration('roboclaw_dry_run')
    roboclaw_usb_port = LaunchConfiguration('roboclaw_usb_port')
    camera_device = LaunchConfiguration('camera_device')
    action_topic = LaunchConfiguration('action_topic')
    use_performance_monitor = LaunchConfiguration('use_performance_monitor')
    performance_output_dir = LaunchConfiguration('performance_output_dir')
    performance_session_name = LaunchConfiguration('performance_session_name')

    default_config = PathJoinSubstitution([
        FindPackageShare('asclinic_vla'),
        'config',
        'asclinic_vla_split.yaml',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'roboclaw_usb_port',
            default_value='/dev/ttyACM0',
            description='RoboClaw serial device.',
        ),
        DeclareLaunchArgument(
            'camera_device',
            default_value='0',
            description='Camera device index, e.g. 0 for /dev/video0 or 1 for /dev/video1.',
        ),
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='YAML config for Jetson hardware split-VLA nodes.',
        ),
        DeclareLaunchArgument(
            'zenoh_connect_endpoint',
            default_value='tcp/127.0.0.1:7447',
            description='Zenoh endpoint for the Brev large VLA/router, e.g. tcp/BREV_IP:7447.',
        ),
        DeclareLaunchArgument(
            'use_goal_publisher',
            default_value='false',
            description='Publish one text/image goal from this launch.',
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
            description='Image path on the Jetson when goal_mode:=image.',
        ),
        DeclareLaunchArgument(
            'roboclaw_dry_run',
            default_value='true',
            description='Run RoboClaw node without driving motors. Set false only when the robot is ready.',
        ),
        DeclareLaunchArgument(
            'action_topic',
            default_value='/asyncvla/action_chunk',
            description='Local ROS ActionChunk topic republished from Zenoh vla/action_chunk.',
        ),
        DeclareLaunchArgument(
            'use_performance_monitor',
            default_value='false',
            description='Record hardware VLA performance CSVs during the run.',
        ),
        DeclareLaunchArgument(
            'performance_output_dir',
            default_value='~/vla_performance_logs',
            description='Directory for hardware performance CSV logs.',
        ),
        DeclareLaunchArgument(
            'performance_session_name',
            default_value='hardware_brev_vla',
            description='Subdirectory/session name for performance logs.',
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
            parameters=[
                config_file,
                {'camera_device': camera_device},
            ],
        ),
        Node(
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
            package='asclinic_vla',
            executable='zenoh_action_chunk_bridge',
            name='zenoh_action_chunk_bridge',
            output='screen',
            parameters=[
                config_file,
                {
                    'action_topic': action_topic,
                    'zenoh_connect_endpoints': [zenoh_connect_endpoint],
                },
            ],
        ),
        Node(
            package='asclinic_vla',
            executable='encoder_odometry',
            name='encoder_odometry',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='asclinic_vla',
            executable='odom_action_chunk_tracker',
            name='odom_action_chunk_tracker',
            output='screen',
            parameters=[
                config_file,
                {'action_topic': action_topic},
            ],
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
                {
                    'dry_run': roboclaw_dry_run,
                    'roboclaw_usb_port': roboclaw_usb_port,
                },
            ],
        ),
        Node(
            package='asclinic_vla',
            executable='vla_performance_hardware',
            name='vla_performance_hardware',
            output='screen',
            parameters=[
                config_file,
                {
                    'mode': 'record',
                    'output_dir': performance_output_dir,
                    'session_name': performance_session_name,
                    'action_topic': action_topic,
                    'zenoh_connect_endpoints': [zenoh_connect_endpoint],
                },
            ],
            condition=IfCondition(use_performance_monitor),
        ),
    ])
