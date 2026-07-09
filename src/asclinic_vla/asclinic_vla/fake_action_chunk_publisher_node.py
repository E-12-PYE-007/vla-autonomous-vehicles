#!/usr/bin/env python3
"""Publish deterministic fake VLA pose chunks for simulation testing.

This node stands in for the AsyncVLA/action-head output while the real model is
unavailable. It publishes robot-relative pose chunks on the same topic used by
the real split VLA path, so the downstream trajectory tracker, wheel reference
logic, and simulator motion path can be tested without changing those nodes.
"""

import math

import rclpy
from asclinic_vla_interfaces.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from rclpy.node import Node


class FakeActionChunkPublisherNode(Node):
    """Generate simple relative pose chunks that look like VLA outputs."""

    def __init__(self):
        super().__init__('fake_action_chunk_publisher')

        self.declare_parameter('action_topic', '/asyncvla/action_chunk')
        self.declare_parameter('publish_rate_hz', 2.0)
        self.declare_parameter('chunk_length', 8)
        self.declare_parameter('step_distance', 0.12)
        self.declare_parameter('turn_per_step', 0.08)
        self.declare_parameter('pattern', 'straight')
        self.declare_parameter('circle_radius', 1.5)
        self.declare_parameter('straight_length', 4.0)

        self.action_topic = self.get_parameter('action_topic').value
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.chunk_length = int(self.get_parameter('chunk_length').value)
        self.step_distance = float(self.get_parameter('step_distance').value)
        self.turn_per_step = float(self.get_parameter('turn_per_step').value)
        self.pattern = str(self.get_parameter('pattern').value).lower()
        self.circle_radius = float(self.get_parameter('circle_radius').value)
        self.straight_length = float(self.get_parameter('straight_length').value)

        self.seq_num = 1
        self.publisher = self.create_publisher(ActionChunk, self.action_topic, 10)
        self.timer = self.create_timer(
            1.0 / max(self.publish_rate_hz, 0.1),
            self.publish_action_chunk,
        )

        self.get_logger().info(
            f'Publishing fake ActionChunk pattern="{self.pattern}" on {self.action_topic}'
        )

    def build_pose(self, index):
        """Return the index-th robot-relative target pose for the test pattern."""
        distance = self.step_distance * index

        if self.pattern == 'stop':
            return 0.0, 0.0, 0.0

        if self.pattern == 'left_arc':
            theta = self.turn_per_step * index
            return distance * math.cos(0.5 * theta), distance * math.sin(0.5 * theta), theta

        if self.pattern == 'right_arc':
            theta = -self.turn_per_step * index
            return distance * math.cos(0.5 * theta), distance * math.sin(0.5 * theta), theta

        if self.pattern == 's_curve':
            theta = self.turn_per_step * math.sin(0.75 * index)
            y = 0.25 * math.sin(0.45 * index)
            return distance, y, theta

        # The default "straight" chunk moves forward in the robot x direction,
        # which matches the relative-pose convention used by the odom tracker.
        return distance, 0.0, 0.0

    def publish_action_chunk(self):
        """Publish one fake chunk in the same message shape as the real model."""
        msg = ActionChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.seq_num = self.seq_num

        for index in range(1, max(self.chunk_length, 1) + 1):
            x, y, theta = self.build_pose(index)
            pose = Pose2D()
            pose.x = float(x)
            pose.y = float(y)
            pose.theta = float(theta)
            msg.relative_poses.append(pose)

        self.publisher.publish(msg)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = FakeActionChunkPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
