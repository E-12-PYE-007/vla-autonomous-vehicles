import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def launch_by_mode(context):
    hardware = LaunchConfiguration('hardware').perform(context).lower()

    if hardware in ('false', '0', 'no'):
        simulation_launch = os.path.join(
            get_package_share_directory('earthrover_vla_simulation'),
            'launch',
            'sim.launch.py',
        )

        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(simulation_launch),
            ),
        ]

    if hardware in ('true', '1', 'yes'):
        return [
            LogInfo(msg="TODO: hardware bringup isn't implemented yet."),
        ]

    return [
        LogInfo(
            msg=(
                "Invalid value for 'hardware': '"
                + hardware
                + "'. Use true or false."
            )
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'hardware',
            default_value='false',
            description='Set to true to request hardware bringup. Defaults to sim.',
        ),
        OpaqueFunction(function=launch_by_mode),
    ])
