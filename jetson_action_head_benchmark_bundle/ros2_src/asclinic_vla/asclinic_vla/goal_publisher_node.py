#!/usr/bin/env python3
"""Publish a text or image goal as `asclinic_vla_interfaces/GoalSpec`."""

import cv2
import rclpy
from asclinic_vla_interfaces.msg import GoalSpec
from rclpy.node import Node
from sensor_msgs.msg import Image


class GoalPublisherNode(Node):
    def __init__(self):
        super().__init__('goal_publisher')

        # This node is intentionally small: it seeds the pipeline with either a
        # language instruction or a reference image, then other nodes decide how
        # to send that goal to the VLA backend.
        self.declare_parameter('goal_topic', '/asyncvla/goal')
        self.declare_parameter('mode', 'text')
        self.declare_parameter('goal_text', '')
        self.declare_parameter('image_path', '')
        self.declare_parameter('publish_period_sec', 1.0)
        self.declare_parameter('publish_once', True)

        self.goal_topic = self.get_parameter('goal_topic').value
        self.mode = self.get_parameter('mode').value
        self.goal_text = self.get_parameter('goal_text').value
        self.image_path = self.get_parameter('image_path').value
        self.publish_once = bool(self.get_parameter('publish_once').value)
        self.published = False

        self.publisher = self.create_publisher(GoalSpec, self.goal_topic, 10)
        period = float(self.get_parameter('publish_period_sec').value)
        self.timer = self.create_timer(max(period, 0.1), self.publish_goal)

        self.get_logger().info(f'Goal publisher ready in {self.mode} mode on {self.goal_topic}')

    @staticmethod
    def cv2_to_ros_image(image) -> Image:
        """Wrap an OpenCV BGR image in a ROS Image message without compression."""
        msg = Image()
        msg.height = image.shape[0]
        msg.width = image.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = image.shape[1] * 3
        msg.data = image.tobytes()
        return msg

    def build_goal_message(self):
        """Create a GoalSpec from the configured mode and parameters."""
        msg = GoalSpec()

        if self.mode == 'text':
            if not self.goal_text:
                self.get_logger().error('mode=text requires a non-empty goal_text parameter')
                return None
            msg.goal_text = self.goal_text
            msg.use_text = True
            msg.use_pose = False
            msg.use_image = False
            return msg

        if self.mode == 'image':
            if not self.image_path:
                self.get_logger().error('mode=image requires an image_path parameter')
                return None
            image = cv2.imread(self.image_path)
            if image is None:
                self.get_logger().error(f'Failed to load goal image: {self.image_path}')
                return None
            msg.goal_image = self.cv2_to_ros_image(image)
            msg.use_text = False
            msg.use_pose = False
            msg.use_image = True
            return msg

        self.get_logger().error(f'Invalid goal mode {self.mode!r}; expected text or image')
        return None

    def publish_goal(self):
        """Publish the goal once by default, or repeatedly if configured."""
        if self.publish_once and self.published:
            return

        msg = self.build_goal_message()
        if msg is None:
            return

        msg.goal_image.header.stamp = self.get_clock().now().to_msg()
        msg.goal_image.header.frame_id = 'goal_image'
        self.publisher.publish(msg)
        self.published = True

        if self.mode == 'text':
            self.get_logger().info(f'Published text goal: {self.goal_text}')
        else:
            self.get_logger().info(f'Published image goal: {self.image_path}')


def main(args=None):
    rclpy.init(args=args)
    node = GoalPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
