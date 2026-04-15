from launch import LaunchDescription

# Import DeclareLaunchArgument so users can pass arguments from the command line
from launch.actions import DeclareLaunchArgument

# Import LaunchConfiguration to access the values of launch arguments
from launch.substitutions import LaunchConfiguration

# Import Node so ROS 2 nodes can be launched from this file
from launch_ros.actions import Node


def generate_launch_description():
    """
    Create and return the full launch description for the AsyncVLA system.

    This launch file starts:
    1. the AsyncVLA inference node
    2. the controller node
    3. the goal publisher node

    It also allows the user to choose whether the goal is text-based
    or image-based using launch arguments.
    """

    # -----------------------------
    # Launch argument values
    # -----------------------------
    # These read the values passed in from the command line at launch time
    goal_mode = LaunchConfiguration('goal_mode')
    goal_text = LaunchConfiguration('goal_text')
    image_path = LaunchConfiguration('image_path')

    # -----------------------------
    # Declare launch arguments
    # -----------------------------
    # Argument for selecting goal mode: either 'text' or 'image'
    declare_goal_mode = DeclareLaunchArgument(
        'goal_mode',
        default_value='text',
        description='Goal mode: text or image'
    )

    # Argument for the text prompt when using text goal mode
    declare_goal_text = DeclareLaunchArgument(
        'goal_text',
        default_value='move forward',
        description='Text goal input'
    )

    # Argument for the image file path when using image goal mode
    declare_image_path = DeclareLaunchArgument(
        'image_path',
        default_value='',
        description='Path to goal image'
    )

    # Return the complete launch description
    return LaunchDescription([

        # -----------------------------
        # Launch arguments
        # -----------------------------
        # These must be declared so they can be used from the terminal
        declare_goal_mode,
        declare_goal_text,
        declare_image_path,

        # -----------------------------
        # AsyncVLA Inference Node
        # -----------------------------
        # This node loads the AsyncVLA model and performs inference
        # using the latest camera images and goal input
        Node(
            package='asyncvla_ros',
            executable='asyncvla_inference_node',
            name='asyncvla_inference_node',
            output='screen',
            parameters=[{
                # Path to the AsyncVLA repository
                'model_repo_path': '/home/miahv/models/AsyncVLA',

                # Path to the trained checkpoint folder
                'checkpoint_path': '/home/miahv/models/AsyncVLA_release',

                # Device used for inference ('cpu' or 'cuda')
                'device': 'cpu',

                # Time between inference runs in seconds
                'inference_period': 0.3
            }]
        ),

        # -----------------------------
        # Controller Node
        # -----------------------------
        # This node converts the predicted ActionChunk trajectory
        # into /cmd_vel velocity commands for the robot
        Node(
            package='asyncvla_ros',
            executable='asyncvla_controller_node',
            name='asyncvla_controller_node',
            output='screen',
            parameters=[{
                # Which predicted trajectory point to follow
                'lookahead_index': 1,

                # Gain for forward velocity
                'k_linear': 0.8,

                # Gain for turning velocity
                'k_angular': 1.5,

                # Maximum forward speed
                'max_linear_speed': 0.25,

                # Maximum turning speed
                'max_angular_speed': 1.0
            }]
        ),

        # -----------------------------
        # Goal Publisher Node
        # -----------------------------
        # This node publishes the goal to /asyncvla/goal.
        # It can publish either a text goal or an image goal,
        # depending on the launch arguments passed in.
        Node(
            package='asyncvla_ros',
            executable='goal_publisher_node',
            name='goal_publisher_node',
            output='screen',
            parameters=[{
                # Goal mode: 'text' or 'image'
                'mode': goal_mode,

                # Text prompt used when goal_mode='text'
                'goal_text': goal_text,

                # Image file path used when goal_mode='image'
                'image_path': image_path
            }]
        ),

    ])