#!/usr/bin/env python3

# Import ROS 2 Python client library
import rclpy
from rclpy.node import Node

# Import the custom goal message and ROS image message type
from asyncvla_interfaces.msg import GoalSpec
from sensor_msgs.msg import Image

# Import OpenCV for image loading and NumPy for array handling
import cv2
import numpy as np


class GoalPublisherNode(Node):
    def __init__(self):
        # Initialize the ROS 2 node with the name 'goal_publisher_node'
        super().__init__('goal_publisher_node')

        # Create a publisher that publishes GoalSpec messages
        # on the topic /asyncvla/goal with queue size 10
        self.publisher = self.create_publisher(
            GoalSpec,
            '/asyncvla/goal',
            10
        )

        # Declare ROS 2 parameters with default values
        # mode determines whether the goal is text-based or image-based
        self.declare_parameter('mode', 'text')      # valid values: 'text' or 'image'
        self.declare_parameter('goal_text', '')     # text prompt for the goal
        self.declare_parameter('image_path', '')    # file path to the goal image

        # Read the parameter values
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.goal_text = self.get_parameter('goal_text').get_parameter_value().string_value
        self.image_path = self.get_parameter('image_path').get_parameter_value().string_value

        # Log which mode the node started in
        self.get_logger().info(f"Goal Publisher started in mode: {self.mode}")

        # Create a timer that calls publish_goal_once() after startup
        # Timer runs every 1 second, but publishing only happens once
        self.timer = self.create_timer(1.0, self.publish_goal_once)

        # Flag to prevent publishing more than once
        self.published = False

    def cv2_to_ros_image(self, img: np.ndarray) -> Image:
        """
        Convert an OpenCV image (NumPy array) into a ROS 2 Image message.
        Assumes the image is in BGR8 format, which is standard for OpenCV.
        """
        msg = Image()
        msg.height = img.shape[0]              # Number of rows
        msg.width = img.shape[1]               # Number of columns
        msg.encoding = 'bgr8'                  # OpenCV default colour format
        msg.is_bigendian = False               # Most systems are little-endian
        msg.step = img.shape[1] * 3            # Number of bytes per row (3 channels)
        msg.data = img.tobytes()               # Convert image array to raw bytes
        return msg

    def publish_goal_once(self):
        """
        Publish the goal message once, based on the selected mode.
        Supports:
        - text mode: publishes a text goal
        - image mode: loads an image from file and publishes it
        """
        # Do nothing if the goal has already been published
        if self.published:
            return

        # Create an empty GoalSpec message
        msg = GoalSpec()

        # ---------- TEXT GOAL MODE ----------
        if self.mode == "text":
            # Check that a text goal was actually provided
            if self.goal_text == "":
                self.get_logger().error("No goal_text provided!")
                return

            # Fill in the message fields for a text goal
            msg.goal_text = self.goal_text
            msg.use_text = True
            msg.use_pose = False
            msg.use_image = False

            # Log what is being published
            self.get_logger().info(f"Publishing TEXT goal: {self.goal_text}")

        # ---------- IMAGE GOAL MODE ----------
        elif self.mode == "image":
            # Check that an image path was provided
            if self.image_path == "":
                self.get_logger().error("No image_path provided!")
                return

            # Load the image from disk using OpenCV
            img = cv2.imread(self.image_path)

            # Check that the image was loaded successfully
            if img is None:
                self.get_logger().error(f"Failed to load image: {self.image_path}")
                return

            # Convert the OpenCV image into a ROS Image message
            ros_img = self.cv2_to_ros_image(img)

            # Fill in the message fields for an image goal
            msg.goal_text = ""
            msg.use_text = False
            msg.use_pose = False
            msg.goal_image = ros_img
            msg.use_image = True

            # Log what is being published
            self.get_logger().info(f"Publishing IMAGE goal: {self.image_path}")

        # ---------- INVALID MODE ----------
        else:
            # Handle invalid mode values
            self.get_logger().error(f"Invalid mode: {self.mode} (use 'text' or 'image')")
            return

        # Publish the message
        self.publisher.publish(msg)

        # Set flag so it only publishes once
        self.published = True

        # Log confirmation
        self.get_logger().info("Goal published successfully")


def main(args=None):
    """
    Main entry point for the ROS 2 node.
    Initializes ROS, creates the node, and keeps it running.
    """
    rclpy.init(args=args)              # Start ROS 2 communication
    node = GoalPublisherNode()         # Create the node
    rclpy.spin(node)                   # Keep the node alive
    node.destroy_node()                # Clean up node resources on shutdown
    rclpy.shutdown()                   # Shut down ROS 2


if __name__ == '__main__':
    main()