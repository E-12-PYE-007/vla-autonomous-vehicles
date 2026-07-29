#!/usr/bin/env python3
"""Publish a text or image goal for the VLA pipeline.

NOTE: original used asclinic_vla_interfaces/GoalSpec which no longer exists.
sys2 currently takes its goal via stdin on startup. This node would need a
new custom message or a std_msgs/String topic wired into sys2 to be useful.
Kept here as a reference for if/when a mid-run goal republishing mechanism
is needed.
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__("goal_publisher")

        self.declare_parameter("goal_topic", "/asyncvla/goal")
        self.declare_parameter("mode", "text")
        self.declare_parameter("goal_text", "")
        self.declare_parameter("image_path", "")
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("publish_once", True)

        self.goal_topic = self.get_parameter("goal_topic").value
        self.mode = self.get_parameter("mode").value
        self.goal_text = self.get_parameter("goal_text").value
        self.image_path = self.get_parameter("image_path").value
        self.publish_once = bool(self.get_parameter("publish_once").value)
        self.published = False

        self.publisher = self.create_publisher(String, self.goal_topic, 10)
        period = float(self.get_parameter("publish_period_sec").value)
        self.timer = self.create_timer(max(period, 0.1), self.publish_goal)

        self.get_logger().info(f"Goal publisher ready in {self.mode} mode on {self.goal_topic}")

    def publish_goal(self):
        """Publish the goal once by default, or repeatedly if configured."""
        if self.publish_once and self.published:
            return

        if self.mode == "text":
            if not self.goal_text:
                self.get_logger().error("mode=text requires a non-empty goal_text parameter")
                return
            msg = String()
            msg.data = self.goal_text
            self.publisher.publish(msg)
            self.published = True
            self.get_logger().info(f"Published text goal: {self.goal_text}")
        else:
            self.get_logger().error(f"Invalid goal mode {self.mode!r}; only text mode supported")


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
