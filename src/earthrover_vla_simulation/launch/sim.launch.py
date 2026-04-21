# ROS2 and Gazebo Launch file for Differential Drive Robot
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.launch_description_sources  import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

def launch_with_custom_world(context):
    # Extract worldfile and name from launch args. Note world name must match worldfile path name.
    worldfile = LaunchConfiguration('worldfile').perform(context)
    world_name = os.path.splitext(os.path.basename(worldfile))[0]

    #Must match robot name in Xacro file
    robotXacroName='earthrover_vla'

    #package names (must match exactly, used to define the paths)
    # description package not required for simulation
    # description_package = 'earthrover_vla_description'
    simulation_package = 'earthrover_vla_simulation'

    #Path to XACRO file defining robot model
    # Note that for simulation, a diferent XACRO is used while hardware uses the standard XACRO in description pkg
    modelFileRealtivePath = 'urdf/robot_sim.xacro'

    # Absolute path to model file
    pathModelFile = os.path.join(get_package_share_directory(simulation_package), modelFileRealtivePath)
    
    #For custom world file relative path
    worldFileRelativePath = f'worlds/{worldfile}'

    # for custom world model
    pathWorldFile = os.path.join(get_package_share_directory(simulation_package),worldFileRelativePath)

    # get robot description from xacro file
    robotDescription = xacro.process_file(pathModelFile).toxml()

    #launch file from the gazebo_ros package
    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ros_gz_sim'),'launch', 'gz_sim.launch.py'))
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPackageLaunch,
        launch_arguments={
            'gz_args': f'-r -v 4 {pathWorldFile}',
            'on_exit_shutdown': 'true'
        }.items()
    )
    # Gazebo node
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', world_name,
            '-name', robotXacroName,
            '-topic', 'robot_description'
        ],
        output = 'screen',
    )

    # Robot state publisher node
    nodeRobotStatePublisher = Node(
        package= 'robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robotDescription,
            'use_sim_time': True}]      
    )

    # To allow control of robot from ROS2
    bridge_params = os.path.join(
        get_package_share_directory(simulation_package),
        'config',
        'bridge_parameters.yaml'
    )

    state_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen'
    )

    # Return all the nodes to be included in the launch obj
    return[
        gazeboLaunch,
        spawnModelNodeGazebo,
        nodeRobotStatePublisher,
        state_gazebo_ros_bridge_cmd
    ]


def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
            'worldfile',
            default_value = 'empty_world_cam.sdf',
            description = 'World file to use when simulating. Path relative to worlds directory. Defaults to empty_world_cam.sdf. Note world name must match path name.',
        ),
        
        OpaqueFunction(function=launch_with_custom_world),
    ])