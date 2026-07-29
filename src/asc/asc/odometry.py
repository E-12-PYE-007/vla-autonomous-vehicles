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


WHEEL_RADIUS = 0.072
WHEEL_BASE = 0.22
ENCODER_COUNTS_PER_REV = 4480
ODOM_FRAME_ID = "odom"
BASE_FRAME_ID = "base_link"


def wrap_to_pi(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class EncoderOdometryNode(Node):
    def __init__(self):
        super().__init__("odometry")

        self.declare_parameter("use_sim", False)
        use_sim = bool(self.get_parameter("use_sim").value)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.pose2d_pub = self.create_publisher(Pose2D, "odom_pose2d", 10)

        if use_sim:
            self.create_subscription(
                Odometry,
                "odom",
                self.sim_odom_callback,
                10,
            )
            self.get_logger().info("Odometry node started in sim mode (passthrough from /odom)")
        else:
            self.create_subscription(
                LeftRightInt32,
                "encoder_counts",
                self.encoder_callback,
                10,
            )
            self.get_logger().info("Encoder odometry node started")

    def sim_odom_callback(self, msg):
        """Pass Gazebo odometry through as odom_pose2d; skip encoder integration."""
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self.theta = wrap_to_pi(2.0 * math.atan2(qz, qw))

        pose = Pose2D()
        pose.x = self.x
        pose.y = self.y
        pose.theta = self.theta
        self.pose2d_pub.publish(pose)

    def encoder_callback(self, msg):
        """Integrate one left/right encoder delta sample."""
        now = self.get_clock().now()
        dt = max((now - self.last_time).nanoseconds / 1e9, 1e-6)
        self.last_time = now

        meters_per_count = (2.0 * math.pi * WHEEL_RADIUS) / ENCODER_COUNTS_PER_REV
        d_left = float(msg.left) * meters_per_count
        d_right = float(msg.right) * meters_per_count

        distance = 0.5 * (d_left + d_right)
        dtheta = (d_right - d_left) / WHEEL_BASE

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
        odom.header.frame_id = ODOM_FRAME_ID
        odom.child_frame_id = BASE_FRAME_ID
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


if __name__ == "__main__":
    main()
