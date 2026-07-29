#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # Camera — publishes ImageWithSeqNum on /cam
            Node(
                package="asc",
                executable="camera_capture",
                name="camera_capture",
                output="screen",
            ),
            # Encoder odometry — reads LeftRightInt32 encoder counts, publishes odom_pose2d
            Node(
                package="asc",
                executable="odometry",
                name="odometry",
                output="screen",
            ),
            # Outer loop — converts ActionChunk + odom_pose2d into wheel_velocity_reference
            Node(
                package="asc",
                executable="outer_loop_controller",
                name="outer_loop_controller",
                output="screen",
            ),
            # Inner loop — wheel velocity PID, outputs set_motor_duty_cycle
            Node(
                package="asc",
                executable="inner_loop_controller",
                name="inner_loop_controller",
                output="screen",
            ),
            # Hardware driver — Roboclaw serial, publishes encoder_counts
            Node(
                package="asc",
                executable="roboclaw_for_motors",
                name="roboclaw_for_motors",
                output="screen",
            ),
        ]
    )
