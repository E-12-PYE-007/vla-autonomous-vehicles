#!/usr/bin/env python3
"""Convert relative VLA action chunks into wheel velocity references.

This node is the open-loop fallback for paths where a VLA backend produces an
ActionChunk directly. The preferred split launch uses `odom_action_chunk_tracker`
so encoder odometry can correct trajectory tracking error.
"""

import math

import rclpy
from asclinic_vla_interfaces.msg import ActionChunk, LeftRightFloat32
from rclpy.node import Node


class ActionChunkPurePursuitNode(Node):
    """Track a VLA robot-frame action chunk and publish wheel velocity targets."""

    def __init__(self):
        super().__init__('action_chunk_pure_pursuit')

        self.declare_parameter('action_topic', '/asyncvla/action_chunk')
        self.declare_parameter('wheel_reference_topic', 'wheel_velocity_reference')
        self.declare_parameter('wheel_base', 0.22)
        self.declare_parameter('lookahead_distance', 0.35)
        self.declare_parameter('max_linear_speed', 0.30)
        self.declare_parameter('max_angular_speed', 1.20)
        self.declare_parameter('command_timeout_sec', 0.75)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('min_forward_target_x', 0.02)

        action_topic = self.get_parameter('action_topic').value
        wheel_reference_topic = self.get_parameter('wheel_reference_topic').value
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.command_timeout_sec = float(self.get_parameter('command_timeout_sec').value)
        self.min_forward_target_x = float(self.get_parameter('min_forward_target_x').value)

        self.last_action = None
        self.last_action_time = None
        self.seq_num = 1

        self.create_subscription(ActionChunk, action_topic, self.action_callback, 10)
        self.publisher = self.create_publisher(LeftRightFloat32, wheel_reference_topic, 10)

        control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(control_rate_hz, 1.0), self.control_loop)
        self.get_logger().info(
            f'Tracking {action_topic}; publishing wheel refs on {wheel_reference_topic}'
        )

    @staticmethod
    def clamp(value, low, high):
        """Clamp a command to configured robot limits."""
        return max(low, min(high, value))

    def action_callback(self, msg):
        """Store the newest short-horizon robot-frame trajectory."""
        if not msg.relative_poses:
            self.get_logger().warn('Ignoring empty VLA action chunk')
            return
        self.last_action = msg
        self.last_action_time = self.get_clock().now()

    def select_lookahead(self):
        """Pick the first pose far enough ahead to act as the pure-pursuit target."""
        poses = self.last_action.relative_poses
        selected = poses[-1]
        for pose in poses:
            if pose.x < self.min_forward_target_x:
                continue
            if math.hypot(pose.x, pose.y) >= self.lookahead_distance:
                selected = pose
                break
        return selected

    def control_loop(self):
        """Run pure pursuit on the latest action chunk and publish wheel refs."""
        if self.last_action is None or self.last_action_time is None:
            self.publish_wheel_refs(0.0, 0.0)
            return

        age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
        if age > self.command_timeout_sec:
            self.publish_wheel_refs(0.0, 0.0)
            return

        target = self.select_lookahead()
        x = float(target.x)
        y = float(target.y)
        distance_sq = max(x * x + y * y, 1e-6)

        if x < self.min_forward_target_x:
            linear = 0.0
        else:
            linear = self.clamp(math.hypot(x, y), 0.0, self.max_linear_speed)

        curvature = 2.0 * y / distance_sq
        angular = self.clamp(linear * curvature, -self.max_angular_speed, self.max_angular_speed)

        # Differential-drive inverse kinematics from body velocity to wheel speeds.
        left = linear - 0.5 * self.wheel_base * angular
        right = linear + 0.5 * self.wheel_base * angular
        self.publish_wheel_refs(left, right)

    def publish_wheel_refs(self, left, right):
        """Publish wheel velocity references consumed by the wheel PID controller."""
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.publisher.publish(msg)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = ActionChunkPurePursuitNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_wheel_refs(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
