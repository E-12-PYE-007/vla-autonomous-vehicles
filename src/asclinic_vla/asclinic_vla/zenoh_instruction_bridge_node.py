#!/usr/bin/env python3
"""Bridge ROS GoalSpec messages into the single Zenoh goal key used by cloud VLA.

The split contract is JSON on `robot/goal`. It carries the same mode flags as
GoalSpec and, for image goals, a JPEG/base64 encoded image. Text and image goals
therefore use one path from ROS to the cloud model.
"""

import base64
import json

import cv2
import numpy as np
import rclpy
from asclinic_vla_interfaces.msg import GoalSpec
from rclpy.node import Node

from asclinic_vla.zenoh_utils import open_zenoh_session


class ZenohInstructionBridgeNode(Node):
    def __init__(self):
        super().__init__('zenoh_instruction_bridge')

        self.declare_parameter('goal_topic', '/asyncvla/goal')
        self.declare_parameter('zenoh_goal_key', 'robot/goal')
        self.declare_parameter('zenoh_legacy_instruction_key', 'robot/instruction')
        self.declare_parameter('publish_legacy_text_instruction', True)
        self.declare_parameter('zenoh_connect_endpoints', 'tcp/127.0.0.1:7447')
        self.declare_parameter('zenoh_listen_endpoints', '')
        self.declare_parameter('republish_duplicates', False)
        self.declare_parameter('image_goal_fallback_text', 'navigate to the goal image')
        self.declare_parameter('jpeg_quality', 90)

        self.goal_key = self.get_parameter('zenoh_goal_key').value
        self.legacy_instruction_key = self.get_parameter('zenoh_legacy_instruction_key').value
        self.publish_legacy_text_instruction = bool(
            self.get_parameter('publish_legacy_text_instruction').value
        )
        self.republish_duplicates = bool(self.get_parameter('republish_duplicates').value)
        self.image_goal_fallback_text = self.get_parameter('image_goal_fallback_text').value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.last_payload = None

        connect_endpoints = self.get_parameter('zenoh_connect_endpoints').value
        listen_endpoints = self.get_parameter('zenoh_listen_endpoints').value

        if isinstance(connect_endpoints, str):
            connect_endpoints = [connect_endpoints] if connect_endpoints else []

        if isinstance(listen_endpoints, str):
            listen_endpoints = [listen_endpoints] if listen_endpoints else []

        self.zenoh_session = open_zenoh_session(
            connect_endpoints,
            listen_endpoints,
        )
        self.goal_publisher = self.zenoh_session.declare_publisher(self.goal_key)
        self.legacy_instruction_publisher = None
        if self.publish_legacy_text_instruction and self.legacy_instruction_key:
            self.legacy_instruction_publisher = self.zenoh_session.declare_publisher(
                self.legacy_instruction_key
            )

        self.create_subscription(
            GoalSpec,
            self.get_parameter('goal_topic').value,
            self.goal_callback,
            10,
        )
        self.get_logger().info(f'Bridging ROS goals to Zenoh key {self.goal_key}')

    @staticmethod
    def ros_image_to_bgr(msg):
        """Convert a GoalSpec image into OpenCV BGR before JPEG encoding."""
        channels = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
        if msg.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if msg.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image[..., :3]

    def encode_goal_image(self, msg):
        """Return a JPEG/base64 object for JSON transport over Zenoh."""
        bgr = self.ros_image_to_bgr(msg)
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError('failed to JPEG encode goal image')
        return {
            'encoding': 'jpeg',
            'data_b64': base64.b64encode(encoded.tobytes()).decode('ascii'),
            'width': int(msg.width),
            'height': int(msg.height),
        }

    def build_goal_payload(self, msg):
        """Convert GoalSpec into the split Zenoh robot/goal JSON contract."""
        payload = {
            't_goal': self.get_clock().now().nanoseconds / 1e9,
            'use_text': bool(msg.use_text),
            'use_image': bool(msg.use_image),
            'use_pose': bool(msg.use_pose),
            'goal_text': msg.goal_text if msg.use_text else '',
        }

        if msg.use_image:
            payload['mode'] = 'image'
            payload['goal_image'] = self.encode_goal_image(msg.goal_image)
            if not payload['goal_text']:
                payload['goal_text'] = self.image_goal_fallback_text
        elif msg.use_text:
            payload['mode'] = 'text'
        else:
            raise ValueError('GoalSpec must set use_text or use_image')

        return payload

    def goal_callback(self, msg):
        """Publish text or image GoalSpec messages over Zenoh."""
        try:
            payload = self.build_goal_payload(msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to build Zenoh goal payload: {exc}')
            return

        payload_json = json.dumps(payload)
        if not self.republish_duplicates and payload_json == self.last_payload:
            return

        self.goal_publisher.put(payload_json.encode('utf-8'))
        if self.legacy_instruction_publisher is not None and payload.get('goal_text'):
            # Compatibility path for the provided AsyncVLA inference/run_vla.py,
            # which subscribes to raw text on `robot/instruction`.
            self.legacy_instruction_publisher.put(payload['goal_text'].encode('utf-8'))
        self.last_payload = payload_json

        self.get_logger().info(
            f"Published Zenoh goal mode={payload['mode']} on {self.goal_key}"
        )

    def destroy_node(self):
        """Release the Zenoh session on shutdown."""
        if hasattr(self, 'zenoh_session'):
            self.zenoh_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZenohInstructionBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
