#!/usr/bin/env python3
"""Convert simulator Odometry into the Pose2D input used by the VLA tracker."""

import math

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quaternion(quaternion):
    """Extract planar yaw from a ROS quaternion."""
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomToPose2DNode(Node):
    """Republish nav_msgs/Odometry as geometry_msgs/Pose2D."""

    def __init__(self):
        super().__init__('odom_to_pose2d')

        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('pose2d_topic', 'odom_pose2d')

        self.publisher = self.create_publisher(
            Pose2D,
            self.get_parameter('pose2d_topic').value,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self.odom_callback,
            10,
        )

        self.get_logger().info(
            f'Converting {self.get_parameter("odom_topic").value} to '
            f'{self.get_parameter("pose2d_topic").value}'
        )

    def odom_callback(self, msg):
        """Publish the planar pose estimate from simulator odometry."""
        pose = Pose2D()
        pose.x = float(msg.pose.pose.position.x)
        pose.y = float(msg.pose.pose.position.y)
        pose.theta = yaw_from_quaternion(msg.pose.pose.orientation)
        self.publisher.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToPose2DNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
