#!/usr/bin/env python3

# OpenCV is used to connect to and read frames from the RTSP camera stream
import cv2

# ROS 2 Python client library
import rclpy
from rclpy.node import Node

# Standard ROS image message type
from sensor_msgs.msg import Image

# Converts between OpenCV images (numpy arrays) and ROS Image messages
from cv_bridge import CvBridge


class RTSPCameraNode(Node):
    def __init__(self):
        # Initialise this ROS 2 node with the name "rtsp_camera_node"
        super().__init__('rtsp_camera_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        # Declare a parameter for the RTSP stream URL.
        # Default is the Earth Rover front camera main stream.
        self.declare_parameter('rtsp_url', 'rtsp://192.168.11.1/live/0')

        # Declare a parameter for how fast images should be published to ROS.
        # Default is 10 Hz.
        self.declare_parameter('publish_rate', 10.0)

        # Read the parameter values
        self.rtsp_url = self.get_parameter('rtsp_url').value
        rate = float(self.get_parameter('publish_rate').value)

        # -----------------------------
        # OpenCV + Bridge
        # -----------------------------
        # Open the RTSP video stream using OpenCV
        self.cap = cv2.VideoCapture(self.rtsp_url)

        # Create a CvBridge object to convert OpenCV frames to ROS Image messages
        self.bridge = CvBridge()

        # Check whether the RTSP stream opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open RTSP stream: {self.rtsp_url}')
        else:
            self.get_logger().info(f'Connected to RTSP stream: {self.rtsp_url}')

        # -----------------------------
        # Publisher
        # -----------------------------
        # Create a ROS publisher that publishes camera frames on /cam
        self.publisher = self.create_publisher(Image, '/cam', 10)

        # -----------------------------
        # Timer
        # -----------------------------
        # Create a timer that calls timer_callback() at the requested publish rate
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def timer_callback(self):
        # Read one frame from the RTSP stream
        ret, frame = self.cap.read()

        # If no frame was received, log a warning and skip this cycle
        if not ret:
            self.get_logger().warn('Failed to grab frame')
            return

        # Convert the OpenCV BGR image into a ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # Add a timestamp so downstream nodes know when this frame was captured/published
        msg.header.stamp = self.get_clock().now().to_msg()

        # Set the frame ID for the image.
        # This is useful later if you build a proper TF/camera frame tree.
        msg.header.frame_id = 'camera_link'

        # Publish the ROS image message on /cam
        self.publisher.publish(msg)


def main(args=None):
    # Start the ROS client library
    rclpy.init(args=args)

    # Create the node
    node = RTSPCameraNode()

    # Keep the node alive so callbacks continue running
    rclpy.spin(node)

    # Clean up the node before shutting down
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()