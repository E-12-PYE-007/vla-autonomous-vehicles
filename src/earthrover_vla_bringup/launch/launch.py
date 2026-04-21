import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_by_mode(context):
    mode = LaunchConfiguration('mode').perform(context).lower()
    worldfile = LaunchConfiguration('worldfile').perform(context)
    
    if mode in ('sim', 'sim-auto-gen'):
        simulation_launch = os.path.join(
            get_package_share_directory('earthrover_vla_simulation'),
            'launch',
            'sim.launch.py',
        )
        #Placeholder branch to generate sdf file. Will set worldfile to the newly generated file.
        if mode == 'sim-auto-gen':
            worldfile=world_gen()

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(simulation_launch),
                launch_arguments={
                    'worldfile': worldfile,
                }.items(),
            ),
        ]

    if mode == 'hardware':
        return [
            LogInfo(msg="TODO: hardware bringup isn't implemented yet."),
        ]
    
    return [
        LogInfo(
            msg= "Invalid mode:"
            + mode
            +"Use hardware, sim, or sim-auto-gen"
        )
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='sim',
            choices = ['hardware', 'sim', 'sim-auto-gen'],
            description='Launch mode: hardware, sim or auto-gen',
        ),

        DeclareLaunchArgument(
            'worldfile',
            default_value = 'empty_world_cam.sdf',
            description = 'World file to use when simulating. Path relative to worlds directory. Defaults to empty_world_cam.sdf. Note world name must match path name.',
        ),
        
        OpaqueFunction(function=launch_by_mode),
    ])

def world_gen():
    # TODO: Implement code to automatically set worldfile, choose 3 random objects and spawn them
    # Will return worldfile
    worldfile = 'empty_office_hallway.sdf'
    return worldfile

