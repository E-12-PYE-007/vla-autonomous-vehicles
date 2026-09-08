#!/usr/bin/env python3
"""Save camera frames from Isaac sim to disk on demand.

Isaac publishes a raw sensor_msgs/Image on /vla/cam (see sys1.py). This node keeps
the latest frame in memory and only writes it to SAVE_DIR when a trigger message is
published on SAVE_TOPIC, so you get one PNG per save command instead of every frame.

Set SAVE_DIR below to whatever folder you want. It's created if it doesn't exist.

Run with:
        ros2 run async_vla save_camera_frames

Then, whenever you want to save the current frame:
        ros2 topic pub --once /save_frame std_msgs/msg/Empty {}
"""

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

# Folder frames are written to. Change this to whatever you want.
SAVE_DIR = "images/fence_left_flat"

# Topic Isaac publishes the raw camera image on.
IMAGE_TOPIC = "/vla/cam"

# Publish any message here to save the latest frame.
SAVE_TOPIC = "/save_frame"


class SaveCameraFramesNode(Node):
    def __init__(self):
        super().__init__("save_camera_frames")

        os.makedirs(SAVE_DIR, exist_ok=True)
        self.bridge = CvBridge()
        self.count = 0
        self.latest = None  # (frame, stamp) of the most recent image

        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 1)
        self.create_subscription(Empty, SAVE_TOPIC, self.save_callback, 1)
        self.get_logger().info(
            f"Buffering frames from {IMAGE_TOPIC}; publish to {SAVE_TOPIC} to save "
            f"to {os.path.abspath(SAVE_DIR)}"
        )

    def image_callback(self, msg: Image):
        # OpenCV writes BGR, so decode straight to bgr8 for correct colours on disk.
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.latest = (frame, stamp)

    def save_callback(self, _msg: Empty):
        if self.latest is None:
            self.get_logger().warn(f"No frame received on {IMAGE_TOPIC} yet, nothing to save")
            return
        frame, stamp = self.latest
        path = os.path.join(SAVE_DIR, f"frame_{self.count:06d}_{stamp}.png")
        cv2.imwrite(path, frame)
        self.count += 1
        self.get_logger().info(f"Saved {path}")


def main(args=None):
    rclpy.init(args=args)
    node = SaveCameraFramesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"Saved {node.count} frames to {os.path.abspath(SAVE_DIR)}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
