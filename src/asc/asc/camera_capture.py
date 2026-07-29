#!/usr/bin/env python3
"""Capture frames from the ASClinic camera and publish them as ROS images."""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from custom_msgs.msg import ImageWithSeqNum

CAMERA_DEVICE = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FPS = 5.0
IMAGE_TOPIC = "/cam"
SHOW_PREVIEW = False
AUTOFOCUS = False
FOCUS = 0
BUFFER_SIZE = 1
VERBOSITY = 1


class CameraCaptureNode(Node):
    def __init__(self):
        super().__init__("camera_capture")

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(ImageWithSeqNum, IMAGE_TOPIC, 10)
        self.seq_num = 0

        self.camera = self._open_camera(CAMERA_DEVICE)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, FPS)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if AUTOFOCUS else 0)
        self.camera.set(cv2.CAP_PROP_FOCUS, FOCUS)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera device: {CAMERA_DEVICE}")

        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0.0 and actual_fps != FPS:
            self.get_logger().warn(f"Camera is running at {actual_fps:.2f} FPS even though {FPS:.2f} FPS was requested")

        timer_fps = actual_fps if actual_fps > 0.0 else FPS
        self.timer = self.create_timer(1.0 / timer_fps, self.publish_frame)
        self.get_logger().info(f"Publishing camera frames from {CAMERA_DEVICE} to {IMAGE_TOPIC} " f"at {FRAME_WIDTH}x{FRAME_HEIGHT}@{timer_fps:.1f} Hz")
        if VERBOSITY >= 1:
            self.get_logger().info("Camera properties: " f"width={self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}, " f"height={self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}, " f"fps={self.camera.get(cv2.CAP_PROP_FPS)}, " f"autofocus={self.camera.get(cv2.CAP_PROP_AUTOFOCUS)}, " f"focus={self.camera.get(cv2.CAP_PROP_FOCUS)}, " f"buffer={self.camera.get(cv2.CAP_PROP_BUFFERSIZE)}")

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
            self.get_logger().warn("Camera read failed")
            return

        msg = ImageWithSeqNum()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.img_seq_num = self.seq_num
        msg.img = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher.publish(msg)
        self.seq_num += 1

        if SHOW_PREVIEW:
            cv2.imshow("asclinic_vla_camera", frame)
            cv2.waitKey(1)

    def destroy_node(self):
        """Release camera resources and preview windows during shutdown."""
        if hasattr(self, "camera"):
            self.camera.release()
        if SHOW_PREVIEW:
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


if __name__ == "__main__":
    main()
