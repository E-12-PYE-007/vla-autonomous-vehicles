#!/usr/bin/env python3
"""Bridge ROS camera frames into the Zenoh payload expected by split AsyncVLA.

The cloud VLA script and Jetson action-head script both subscribe to the
`camera/img_compressed` Zenoh key. They expect JSON containing two JPEG byte
strings encoded with latin-1: `past_img` and `curr_img`. This node subscribes
to a normal ROS `sensor_msgs/Image`, keeps the previous frame, JPEG-compresses
both frames, and publishes that JSON payload over Zenoh.
"""

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from asclinic_vla.zenoh_utils import open_zenoh_session


class ZenohCameraBridgeNode(Node):
    def __init__(self):
        super().__init__('zenoh_camera_bridge')

        # ROS input topic plus Zenoh key/endpoint settings are launch-configurable
        # so the same node can run against localhost, a Zenoh router, or the cloud.
        self.declare_parameter('image_topic', '/cam')
        self.declare_parameter('zenoh_camera_key', 'camera/img_compressed')
        self.declare_parameter('zenoh_connect_endpoints', 'tcp/127.0.0.1:7447')
        self.declare_parameter('zenoh_listen_endpoints', '')
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('publish_rate_hz', 5.0)

        self.zenoh_key = self.get_parameter('zenoh_camera_key').value
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.min_publish_period = 0.0 if publish_rate_hz <= 0.0 else 1.0 / publish_rate_hz
        self.last_publish_time = 0.0
        self.prev_jpeg = None
        self.curr_jpeg = None
        self.frame_id = 0

        # Open the Zenoh session before subscribing so camera callbacks can publish
        # immediately once frames start arriving.
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
        self.publisher = self.zenoh_session.declare_publisher(self.zenoh_key)

        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f'Bridging ROS camera frames to Zenoh key {self.zenoh_key}')

    @staticmethod
    def ros_image_to_bgr(msg):
        """Convert common ROS image encodings into OpenCV BGR for JPEG encoding."""
        channels = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, channels))
        if msg.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if msg.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if msg.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image[..., :3]

    def image_callback(self, msg):
        """Throttle, compress, and publish previous/current camera frames to Zenoh."""
        now = time.time()
        if now - self.last_publish_time < self.min_publish_period:
            return

        bgr = self.ros_image_to_bgr(msg)
        ok, encoded = cv2.imencode(
            '.jpg',
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            self.get_logger().warn('Failed to JPEG-encode camera frame')
            return

        self.prev_jpeg = self.curr_jpeg
        self.curr_jpeg = encoded.tobytes()
        # The action head needs temporal context, so wait until we have two frames.
        if self.prev_jpeg is None:
            return

        # JSON cannot carry raw bytes directly. latin-1 preserves every byte value
        # one-to-one and matches the provided run_vla.py/run_action_head.py scripts.
        payload = {
            'frame_id': self.frame_id,
            't_camera': now,
            'past_img': self.prev_jpeg.decode('latin-1'),
            'curr_img': self.curr_jpeg.decode('latin-1'),
        }
        self.publisher.put(json.dumps(payload).encode('utf-8'))
        self.frame_id += 1
        self.last_publish_time = now

    def destroy_node(self):
        """Close Zenoh cleanly when ROS shuts the node down."""
        if hasattr(self, 'zenoh_session'):
            self.zenoh_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZenohCameraBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
