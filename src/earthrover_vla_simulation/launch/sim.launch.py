# ROS2 and Gazebo Launch file for Differential Drive Robot
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources  import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro

def generate_launch_description():
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
    worldName = 'empty_world_cam'
    worldFileRelativePath = f'worlds/{worldName}.sdf'

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
            '-world', 'empty_world_cam',
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

    # create an empty launch description object
    launchDescriptionObject = LaunchDescription()
    
    # add gazebolaunch
    launchDescriptionObject.add_action(gazeboLaunch)

    # add the two nodes
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(state_gazebo_ros_bridge_cmd)

    return launchDescriptionObject
