#!/usr/bin/env python3
"""Simulator + asc control nodes. No inference.

Normally you do not launch this directly — the per-stack sim launches include it and
pick the matching controller for you:

    ros2 launch async_vla sim_async.launch.py
    ros2 launch tic_vla   sim_tic.launch.py

Launch it on its own only for a split setup, with inference on another machine:

    VM:       ros2 launch asc asc_sim.launch.py controller:=outer_loop_controller
    desktop:  ros2 launch async_vla asc_async.launch.py device:=rcp

Two simulators are supported:

    sim:=gazebo   Starts Gazebo through earthrover_vla_bringup, which already runs
                  cam_seq_bridge, so /cam carries ImageWithSeqNum.
    sim:=isaac    Starts nothing simulator-side; Isaac runs separately. Supplies
                  cam_seq_bridge here instead, to turn the raw sensor_msgs/Image on
                  /cam_raw into the ImageWithSeqNum the inference nodes subscribe to.
                  Without it they start cleanly, log nothing unusual, and never receive
                  a frame. Configure Isaac to publish the camera on /cam_raw, the same
                  topic the Gazebo bridge uses.

Whichever simulator is used, it must publish nav_msgs/Odometry (on /odom for gazebo,
/sim_odom for isaac) and subscribe geometry_msgs/Twist on /cmd_vel.

When pairing with inference over an SSH tunnel, RMW_IMPLEMENTATION and ROS_DOMAIN_ID must
match on both machines — set them in the shell, see the README networking section. A local
one-machine run needs neither.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Edit here to run a different world. Deliberately not a launch argument: it is changed
# rarely, and every argument threaded through the per-stack sim launches is wiring that
# has to be kept correct in three files.
WORLDFILE = "unempty_office_square.sdf"


def launch_setup(context, *args, **kwargs):
    sim = LaunchConfiguration("sim").perform(context).lower()

    # Gazebo publishes odometry on /odom, Isaac on /sim_odom. Derived from `sim` rather
    # than exposed as another argument; the odometry node republishes onto /odom when it
    # differs, so nothing downstream needs to know which simulator is running.
    odom_topic = "/sim_odom" if sim == "isaac" else "/odom"

    if sim == "gazebo":
        earthrover_bringup_launch = os.path.join(
            get_package_share_directory("earthrover_vla_bringup"),
            "launch",
            "launch.py",
        )
        simulator = [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(earthrover_bringup_launch),
                launch_arguments={
                    "mode": "sim",
                    "worldfile": WORLDFILE,
                }.items(),
            ),
        ]
    else:
        simulator = [
            Node(
                package="earthrover_vla_simulation",
                executable="cam_seq_bridge.py",
                name="cam_seq_bridge",
                output="screen",
            ),
        ]

    control = [
        # Republishes the simulator's /odom as odom_pose2d. In sim mode this is a
        # passthrough; it does not integrate encoders.
        Node(
            package="asc",
            executable="odometry",
            name="encoder_odometry",
            output="screen",
            parameters=[{"use_sim": True}, {"odom_topic": odom_topic}],
        ),
        # Turns action chunks into /cmd_vel. The controllers are not interchangeable —
        # see the controller argument.
        Node(
            package="asc",
            executable=LaunchConfiguration("controller"),
            name="controller",
            output="screen",
            parameters=[{"use_sim": True}],
        ),
    ]

    return simulator + control


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sim",
                default_value="gazebo",
                choices=["gazebo", "isaac"],
                description="Which simulator to pair with. gazebo is started here; "
                            "isaac runs separately and only needs the camera bridge.",
            ),
            # The controllers are not interchangeable: tic_controller is a port of
            # TIC-VLA's benchmark driver (V_MAX 0.30, 1.0 m arc-length lookahead, bearing
            # feedback + curvature feedforward) tuned for a model trained at ~1.5 m/s
            # emitting 30-step plans. Driving AsyncVLA's shorter chunks with it oversteers
            # badly. The per-stack sim launches set this for you.
            DeclareLaunchArgument(
                "controller",
                default_value="outer_loop_controller",
                description="asc controller executable: outer_loop_controller (AsyncVLA) | "
                            "tic_controller (TIC-VLA benchmark port) | pure_pursuit_controller.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
