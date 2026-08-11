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
from custom_msgs.msg import ImageWithSeqNum, TicPixelValues
from ticvla.utils.vision import build_transform, dynamic_preprocess

IMAGE_INPUT_SIZE = 448


class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__("tic_vla_image_processing")

        self.bridge = CvBridge()
        self.transform = build_transform(input_size=IMAGE_INPUT_SIZE)

        self.create_subscription(ImageWithSeqNum, "/cam", self.image_callback, 1)
        self.pub = self.create_publisher(
            TicPixelValues,
            "/tic_vla/pixel_values",
            1,
        )
        self.get_logger().info(f"Preprocessing /cam → /tic_vla/pixel_values at {IMAGE_INPUT_SIZE}px")

    def image_callback(self, msg: ImageWithSeqNum):
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg.img, desired_encoding="rgb8")
        pil_img = PILImage.fromarray(cv_img)
        # max_num=1 → a single whole-frame tile (NUM_IMAGE_TOKEN visual tokens), matching
        # upstream inference, which uses load_image(..., max_num=1) for both the delayed
        # frames fed to the VLM and the current frame fed to the ActionExpert.
        tiles = dynamic_preprocess(pil_img, image_size=IMAGE_INPUT_SIZE, use_thumbnail=True, max_num=1)
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
