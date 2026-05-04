#!/usr/bin/env python3
"""Capture frames from the ASClinic camera and publish them as ROS images."""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraCaptureNode(Node):
    def __init__(self):
        super().__init__('asclinic_camera_capture')

        # Keep camera settings configurable because USB camera indices and supported
        # resolutions often differ between the laptop, Jetson, and final robot.
        self.declare_parameter('camera_device', 0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('image_topic', '/cam')
        self.declare_parameter('show_preview', False)

        camera_device = self.get_parameter('camera_device').value
        frame_width = int(self.get_parameter('frame_width').value)
        frame_height = int(self.get_parameter('frame_height').value)
        fps = float(self.get_parameter('fps').value)
        image_topic = self.get_parameter('image_topic').value
        self.show_preview = bool(self.get_parameter('show_preview').value)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, image_topic, 10)

        # OpenCV is used here rather than a camera-specific ROS driver so the
        # project can start with the same simple camera path as the ASClinic code.
        self.camera = self._open_camera(camera_device)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.camera.isOpened():
            raise RuntimeError(f'Failed to open camera device: {camera_device}')

        self.timer = self.create_timer(1.0 / max(fps, 1.0), self.publish_frame)
        self.get_logger().info(
            f'Publishing camera frames from {camera_device} to {image_topic} '
            f'at {frame_width}x{frame_height}@{fps:.1f} Hz'
        )

    def _open_camera(self, camera_device):
        """Open numeric devices with V4L2, or string pipelines/paths directly."""
        try:
            return cv2.VideoCapture(int(camera_device), cv2.CAP_V4L2)
        except (TypeError, ValueError):
            return cv2.VideoCapture(str(camera_device))

    def publish_frame(self):
        """Read one frame, stamp it, and publish it on the configured ROS topic."""
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.get_logger().warn('Camera read failed')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.publisher.publish(msg)

        if self.show_preview:
            cv2.imshow('asclinic_vla_camera', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        """Release camera resources and preview windows during shutdown."""
        if hasattr(self, 'camera'):
            self.camera.release()
        if getattr(self, 'show_preview', False):
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraCaptureNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
