#!/usr/bin/env python3
"""Estimate robot pose by integrating left/right encoder count deltas.

The Roboclaw node publishes encoder deltas for each control interval. This node
converts those deltas to wheel travel, integrates differential-drive odometry,
and publishes both a standard `nav_msgs/Odometry` message and a compact
`geometry_msgs/Pose2D` message for controllers that only need planar pose.
"""

import math

import rclpy
from custom_msgs.msg import LeftRightInt32
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node


def wrap_to_pi(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class EncoderOdometryNode(Node):
    def __init__(self):
        super().__init__('encoder_odometry')

        self.declare_parameter('encoder_counts_topic', 'encoder_counts')
        self.declare_parameter('odom_topic', 'odom')
        self.declare_parameter('pose2d_topic', 'odom_pose2d')
        self.declare_parameter('wheel_radius', 0.072)
        self.declare_parameter('wheel_base', 0.22)
        self.declare_parameter('encoder_counts_per_rev', 4480)
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')

        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.counts_per_rev = int(self.get_parameter('encoder_counts_per_rev').value)
        self.odom_frame_id = self.get_parameter('odom_frame_id').value
        self.base_frame_id = self.get_parameter('base_frame_id').value

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, self.get_parameter('odom_topic').value, 10)
        self.pose2d_pub = self.create_publisher(Pose2D, self.get_parameter('pose2d_topic').value, 10)

        self.create_subscription(
            LeftRightInt32,
            self.get_parameter('encoder_counts_topic').value,
            self.encoder_callback,
            10,
        )

        self.get_logger().info('Encoder odometry node started')

    def encoder_callback(self, msg):
        """Integrate one left/right encoder delta sample."""
        now = self.get_clock().now()
        dt = max((now - self.last_time).nanoseconds / 1e9, 1e-6)
        self.last_time = now

        meters_per_count = (2.0 * math.pi * self.wheel_radius) / self.counts_per_rev
        d_left = float(msg.left) * meters_per_count
        d_right = float(msg.right) * meters_per_count

        distance = 0.5 * (d_left + d_right)
        dtheta = (d_right - d_left) / self.wheel_base

        # Midpoint integration is more accurate than using the old heading for
        # the whole interval, especially while turning.
        theta_mid = self.theta + 0.5 * dtheta
        self.x += distance * math.cos(theta_mid)
        self.y += distance * math.sin(theta_mid)
        self.theta = wrap_to_pi(self.theta + dtheta)

        linear_velocity = distance / dt
        angular_velocity = dtheta / dt
        self.publish_pose(now, linear_velocity, angular_velocity)

    def publish_pose(self, stamp, linear_velocity, angular_velocity):
        """Publish the current planar pose as Odometry and Pose2D."""
        odom = Odometry()
        odom.header.stamp = stamp.to_msg()
        odom.header.frame_id = self.odom_frame_id
        odom.child_frame_id = self.base_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = math.sin(0.5 * self.theta)
        odom.pose.pose.orientation.w = math.cos(0.5 * self.theta)
        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.angular.z = angular_velocity
        self.odom_pub.publish(odom)

        pose = Pose2D()
        pose.x = self.x
        pose.y = self.y
        pose.theta = self.theta
        self.pose2d_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = EncoderOdometryNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
