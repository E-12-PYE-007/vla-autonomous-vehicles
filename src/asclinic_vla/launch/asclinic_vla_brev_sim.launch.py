from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    worldfile = LaunchConfiguration('worldfile')
    goal_text = LaunchConfiguration('goal_text')
    zenoh_endpoint = LaunchConfiguration('zenoh_endpoint')
    angular_multiplier = LaunchConfiguration('angular_multiplier')
    linear_multiplier = LaunchConfiguration('linear_multiplier')
    goal_publish_delay = LaunchConfiguration('goal_publish_delay')

    sim_launch = PathJoinSubstitution([
        FindPackageShare('earthrover_vla_simulation'),
        'launch',
        'sim.launch.py',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'worldfile',
            default_value='unempty_office_square.sdf',
            description='Gazebo world file under earthrover_vla_simulation/worlds/templates.',
        ),
        DeclareLaunchArgument(
            'goal_text',
            default_value='go to box',
            description='Text goal sent to AsyncVLA.',
        ),
        DeclareLaunchArgument(
            'zenoh_endpoint',
            default_value='tcp/127.0.0.1:7447',
            description='Local Zenoh endpoint, usually the Brev port-forward.',
        ),
        DeclareLaunchArgument(
            'angular_multiplier',
            default_value='-1.0',
            description='Use -1.0 when AsyncVLA steering is opposite ROS/Gazebo angular.z.',
        ),
        DeclareLaunchArgument(
            'linear_multiplier',
            default_value='1.0',
            description='Use -1.0 only if forward/backward is reversed.',
        ),
        DeclareLaunchArgument(
            'goal_publish_delay',
            default_value='4.0',
            description='Seconds to wait before publishing the one-shot goal.',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch),
            launch_arguments={'worldfile': worldfile}.items(),
        ),

        Node(
            package='asclinic_vla',
            executable='zenoh_camera_bridge',
            name='zenoh_camera_bridge',
            output='screen',
            parameters=[{
                'image_topic': '/cam',
                'zenoh_camera_key': 'camera/img_compressed',
                'zenoh_connect_endpoints': [zenoh_endpoint],
            }],
        ),
        Node(
            package='asclinic_vla',
            executable='zenoh_instruction_bridge',
            name='zenoh_instruction_bridge',
            output='screen',
            parameters=[{
                'goal_topic': '/asyncvla/goal',
                'zenoh_goal_key': 'robot/goal',
                'zenoh_legacy_instruction_key': 'robot/instruction',
                'publish_legacy_text_instruction': True,
                'zenoh_connect_endpoints': [zenoh_endpoint],
            }],
        ),
        TimerAction(
            period=goal_publish_delay,
            actions=[
                Node(
                    package='asclinic_vla',
                    executable='goal_publisher',
                    name='goal_publisher',
                    output='screen',
                    parameters=[{
                        'mode': 'text',
                        'goal_text': goal_text,
                        'publish_once': True,
                    }],
                ),
            ],
        ),
        Node(
            package='asclinic_vla',
            executable='zenoh_cmd_vel_bridge',
            name='zenoh_cmd_vel_bridge',
            output='screen',
            parameters=[{
                'zenoh_cmd_vel_key': 'vla/cmd_vel',
                'wheel_reference_topic': 'wheel_velocity_reference',
                'zenoh_connect_endpoints': [zenoh_endpoint],
                'linear_multiplier': linear_multiplier,
                'angular_multiplier': angular_multiplier,
            }],
        ),
        Node(
            package='asclinic_vla',
            executable='wheel_reference_to_cmd_vel',
            name='wheel_reference_to_cmd_vel',
            output='screen',
            parameters=[{
                'wheel_reference_topic': 'wheel_velocity_reference',
                'cmd_vel_topic': '/cmd_vel',
            }],
        ),
    ])
