#!/usr/bin/env python3
"""Optional direct HTTP VLA client kept for non-Zenoh experiments.

The main split architecture uses Zenoh and the Jetson action head. This node is
still useful for testing a simpler remote API that returns ActionChunk messages
directly, bypassing the split action-head path.
"""

import base64
import io

import numpy as np
import rclpy
import requests
from asclinic_vla_interfaces.msg import ActionChunk, GoalSpec
from geometry_msgs.msg import Pose2D
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage


class VLAHttpClientNode(Node):
    def __init__(self):
        super().__init__('vla_http_client')

        # All topics and timing values are parameters so this client can be used
        # in simulation, on the robot, or against a mock HTTP endpoint.
        self.declare_parameter('image_topic', '/cam')
        self.declare_parameter('goal_topic', '/asyncvla/goal')
        self.declare_parameter('action_topic', '/asyncvla/action_chunk')
        self.declare_parameter('api_url', 'http://127.0.0.1:8000/predict')
        self.declare_parameter('inference_rate_hz', 2.0)
        self.declare_parameter('request_timeout_sec', 10.0)

        self.api_url = self.get_parameter('api_url').value
        self.request_timeout_sec = float(self.get_parameter('request_timeout_sec').value)
        self.prev_image = None
        self.curr_image = None
        self.goal_text = ''
        self.goal_image = None
        self.use_text = False
        self.use_image = False

        self.create_subscription(
            ROSImage,
            self.get_parameter('image_topic').value,
            self.image_callback,
            10,
        )
        self.create_subscription(
            GoalSpec,
            self.get_parameter('goal_topic').value,
            self.goal_callback,
            10,
        )
        self.action_publisher = self.create_publisher(
            ActionChunk,
            self.get_parameter('action_topic').value,
            10,
        )

        inference_rate_hz = float(self.get_parameter('inference_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(inference_rate_hz, 0.1), self.run_inference)
        self.get_logger().info(f'VLA HTTP client ready: {self.api_url}')

    def image_callback(self, msg):
        """Store the previous/current frame pair expected by AsyncVLA."""
        self.prev_image = self.curr_image
        self.curr_image = msg

    def goal_callback(self, msg):
        """Store the latest text or image goal from ROS."""
        self.use_text = bool(msg.use_text)
        self.use_image = bool(msg.use_image)
        self.goal_text = msg.goal_text if self.use_text else ''
        self.goal_image = msg.goal_image if self.use_image else None

    @staticmethod
    def ros_image_to_rgb(msg):
        """Convert ROS image bytes into an RGB NumPy array for PIL encoding."""
        channels = 3
        if msg.encoding in ('rgba8', 'bgra8'):
            channels = 4
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
        if msg.encoding in ('bgr8', 'bgra8'):
            image = image[..., :3][..., ::-1]
        else:
            image = image[..., :3]
        return image

    @staticmethod
    def encode_png_base64(image):
        """Encode an RGB image as base64 PNG for JSON HTTP transport."""
        buffer = io.BytesIO()
        PILImage.fromarray(image).save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('ascii')

    def run_inference(self):
        """Call the HTTP VLA API and republish its relative poses as ActionChunk."""
        if self.prev_image is None or self.curr_image is None:
            return
        if not self.use_text and not self.use_image:
            return

        payload = {
            'past_image_b64': self.encode_png_base64(self.ros_image_to_rgb(self.prev_image)),
            'current_image_b64': self.encode_png_base64(self.ros_image_to_rgb(self.curr_image)),
        }
        if self.use_text:
            payload['goal_text'] = self.goal_text
        if self.use_image and self.goal_image is not None:
            payload['goal_image_b64'] = self.encode_png_base64(self.ros_image_to_rgb(self.goal_image))

        try:
            response = requests.post(self.api_url, json=payload, timeout=self.request_timeout_sec)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            self.get_logger().error(f'VLA HTTP request failed: {exc}')
            return

        relative_poses = data.get('relative_poses', [])
        if not relative_poses:
            self.get_logger().warn('VLA response did not include relative_poses')
            return

        msg = ActionChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, theta in relative_poses:
            pose = Pose2D()
            pose.x = float(x)
            pose.y = float(y)
            pose.theta = float(theta)
            msg.relative_poses.append(pose)
        self.action_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLAHttpClientNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
