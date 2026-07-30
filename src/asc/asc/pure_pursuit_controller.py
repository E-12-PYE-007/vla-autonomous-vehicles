#!/usr/bin/env python3
"""Open-loop pure pursuit controller for VLA action chunks.

Simpler alternative to outer_loop_controller — no odometry feedback,
tracks directly in robot frame. Use when you don't need cross-track correction.
"""

import math

import rclpy
from custom_msgs.msg import ActionChunk, LeftRightFloat32
from geometry_msgs.msg import Twist
from rclpy.node import Node


WHEEL_BASE = 0.22
LOOKAHEAD_DISTANCE = 0.35
MAX_LINEAR_SPEED = 0.30
MAX_ANGULAR_SPEED = 1.20
COMMAND_TIMEOUT_SEC = 0.75
CONTROL_RATE_HZ = 10.0
MIN_FORWARD_TARGET_X = 0.02


class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit_controller")

        self.declare_parameter("use_sim", False)
        self.use_sim = bool(self.get_parameter("use_sim").value)

        self.last_action = None
        self.last_action_time = None
        self.seq_num = 1

        self.create_subscription(ActionChunk, "/asyncvla/action_chunk", self.action_callback, 10)
        self.create_subscription(ActionChunk, "/ticvla/action_chunk", self.action_callback, 10)

        if self.use_sim:
            self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        else:
            self.publisher = self.create_publisher(LeftRightFloat32, "wheel_velocity_reference", 10)

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.get_logger().info("Pure pursuit controller started")

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def action_callback(self, msg):
        if not msg.relative_poses:
            self.get_logger().warn("Ignoring empty VLA action chunk")
            return
        self.last_action = msg
        self.last_action_time = self.get_clock().now()

    def select_lookahead(self):
        """Pick the first pose far enough ahead as the pure-pursuit target."""
        poses = self.last_action.relative_poses
        selected = poses[-1]
        for pose in poses:
            if pose.x < MIN_FORWARD_TARGET_X:
                continue
            if math.hypot(pose.x, pose.y) >= LOOKAHEAD_DISTANCE:
                selected = pose
                break
        return selected

    def control_loop(self):
        if self.last_action is None or self.last_action_time is None:
            self.publish_wheel_refs(0.0, 0.0)
            return

        age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
        if age > COMMAND_TIMEOUT_SEC:
            self.publish_wheel_refs(0.0, 0.0)
            return

        target = self.select_lookahead()
        x = float(target.x)
        y = float(target.y)
        distance_sq = max(x * x + y * y, 1e-6)

        if x < MIN_FORWARD_TARGET_X:
            linear = 0.0
        else:
            linear = self.clamp(math.hypot(x, y), 0.0, MAX_LINEAR_SPEED)

        curvature = 2.0 * y / distance_sq
        angular = self.clamp(linear * curvature, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        left = linear - 0.5 * WHEEL_BASE * angular
        right = linear + 0.5 * WHEEL_BASE * angular
        self.publish_wheel_refs(left, right)

    def publish_wheel_refs(self, left, right):
        if self.use_sim:
            msg = Twist()
            msg.linear.x = (left + right) / 2.0
            msg.angular.z = (right - left) / WHEEL_BASE
            self.publisher.publish(msg)
        else:
            msg = LeftRightFloat32()
            msg.left = float(left)
            msg.right = float(right)
            msg.seq_num = self.seq_num
            self.publisher.publish(msg)
            self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_wheel_refs(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
