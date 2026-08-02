#!/usr/bin/env python3
"""Preprocess raw camera frames for TIC-VLA and publish as pixel value tensors.

Runs dynamic_preprocess + build_transform (following ticvla/utils/vision.py load_image)
and publishes the resulting (num_tiles, 3, 448, 448) float32 tensor as TicPixelValues.
Both sys1 (fast encode path) and sys2 (slow VLM path) subscribe to this topic.
"""

import torch
from PIL import Image as PILImage

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from custom_msgs.msg import ImageWithSeqNum, TicPixelValues
from ticvla.utils.vision import build_transform, dynamic_preprocess

IMAGE_INPUT_SIZE = 448


class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__("tic_vla_image_processing")

        self.bridge = CvBridge()
        self.transform = build_transform(input_size=IMAGE_INPUT_SIZE)
        self._sim_seq_num = 0

        self.declare_parameter("use_sim", False)
        use_sim = bool(self.get_parameter("use_sim").value)

        if use_sim:
            self.create_subscription(Image, "/cam", self.sim_image_callback, 1)
        else:
            self.create_subscription(ImageWithSeqNum, "/cam", self.image_callback, 1)
        self.pub = self.create_publisher(
            TicPixelValues,
            "/tic_vla/pixel_values",
            1,
        )
        self.get_logger().info(f"Preprocessing /cam → /tic_vla/pixel_values at {IMAGE_INPUT_SIZE}px")

    def sim_image_callback(self, msg: Image):
        wrapped = ImageWithSeqNum()
        wrapped.header = msg.header
        wrapped.img = msg
        wrapped.img_seq_num = self._sim_seq_num
        self._sim_seq_num += 1
        self.image_callback(wrapped)

    def image_callback(self, msg: ImageWithSeqNum):
        cv_img = self.bridge.imgmsg_to_cv2(msg.img, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img)
        tiles = dynamic_preprocess(pil_img, image_size=IMAGE_INPUT_SIZE, use_thumbnail=True, max_num=12)
        pixel_values = torch.stack([self.transform(t) for t in tiles])  # (N, 3, H, W) float32

        out = TicPixelValues()
        out.header = msg.header
        out.img_seq_num = msg.img_seq_num
        out.num_tiles = len(tiles)
        out.data.data = pixel_values.flatten().tolist()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
