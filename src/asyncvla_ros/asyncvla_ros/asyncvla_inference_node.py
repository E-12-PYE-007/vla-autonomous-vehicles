#!/usr/bin/env python3

# Enables postponed evaluation of type hints
from __future__ import annotations

# NumPy is used for image array conversion and manipulation
import numpy as np

# ROS 2 Python client library
import rclpy
from rclpy.node import Node

# ROS message types
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from asyncvla_interfaces.msg import GoalSpec, ActionChunk

# Backend class that handles loading the AsyncVLA model and running inference
from asyncvla_ros.asyncvla_backend import AsyncVLABackend


class AsyncVLAInferenceNode(Node):
    def __init__(self):
        # Initialize this ROS 2 node
        super().__init__('asyncvla_inference_node')

        # -----------------------------
        # Internal state
        # -----------------------------
        # Store the previous and latest camera image messages
        self.prev_image_msg = None
        self.latest_image_msg = None

        # Store the most recent goal message
        self.latest_goal_msg = None

        # Flag to prevent starting a new inference while one is already running
        self.inference_running = False

        # -----------------------------
        # Parameters
        # -----------------------------
        # Path to AsyncVLA model repo
        self.declare_parameter('model_repo_path', '')

        # Path to trained checkpoint
        self.declare_parameter('checkpoint_path', '')

        # Device to run inference on, e.g. 'cuda' or 'cpu'
        self.declare_parameter('device', 'cuda')

        # Time between inference calls in seconds
        self.declare_parameter('inference_period', 0.2)

        # Read parameter values
        model_repo_path = self.get_parameter('model_repo_path').value
        checkpoint_path = self.get_parameter('checkpoint_path').value
        device = self.get_parameter('device').value
        self.inference_period = float(self.get_parameter('inference_period').value)

        # -----------------------------
        # Backend
        # -----------------------------
        # Create backend object that wraps model loading and prediction
        self.backend = AsyncVLABackend(
            model_repo_path=model_repo_path or None,
            checkpoint_path=checkpoint_path or None,
            device=device,
        )

        # Load the AsyncVLA model into memory
        self.backend.load_model()

        # -----------------------------
        # Subscribers
        # -----------------------------
        # Subscribe to camera images
        self.image_sub = self.create_subscription(
            Image,
            '/cam',
            self.image_callback,
            10
        )

        # Subscribe to the goal topic
        self.goal_sub = self.create_subscription(
            GoalSpec,
            '/asyncvla/goal',
            self.goal_callback,
            10
        )

        # -----------------------------
        # Publisher
        # -----------------------------
        # Publish predicted action chunks
        self.action_pub = self.create_publisher(
            ActionChunk,
            '/asyncvla/action_chunk',
            10
        )

        # -----------------------------
        # Timer
        # -----------------------------
        # Run inference periodically
        self.timer = self.create_timer(
            self.inference_period,
            self.timer_callback
        )

        # Log startup info
        self.get_logger().info('AsyncVLA Inference Node started')
        self.get_logger().info(f'Inference period: {self.inference_period}s')

    # =========================
    # Callbacks
    # =========================

    def image_callback(self, msg: Image):
        # Shift the current image to previous, then store the new image
        # This lets the model use both past and current frames
        self.prev_image_msg = self.latest_image_msg
        self.latest_image_msg = msg

    def goal_callback(self, msg: GoalSpec):
        # Save the latest received goal
        self.latest_goal_msg = msg

        # Print goal details for debugging
        self.get_logger().info(
            f"Received goal: text='{msg.goal_text}', "
            f"use_text={msg.use_text}, use_pose={msg.use_pose}, use_image={msg.use_image}"
        )

    # =========================
    # Utilities
    # =========================

    def ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        """
        Convert a ROS Image message into a NumPy array.
        Assumes the image is RGB/BGR with 3 channels.
        """
        # Check image dimensions are valid
        if msg.height == 0 or msg.width == 0:
            raise ValueError('Empty image received')

        channels = 3
        expected_size = msg.height * msg.width * channels

        # Convert raw byte data into a NumPy array
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        # Check that enough image data exists
        if raw.size < expected_size:
            raise ValueError(f'Image data too small: {raw.size} < {expected_size}')

        # Reshape flat array into (height, width, channels)
        return raw[:expected_size].reshape((msg.height, msg.width, channels))

    def extract_goal_pose(self, msg: GoalSpec):
        """
        Extract (x, y, theta) from the goal message if pose mode is enabled.
        Currently theta is set to 0.0.
        """
        if not msg.use_pose:
            return None

        pose = msg.target_pose.pose
        return (pose.position.x, pose.position.y, 0.0)

    # =========================
    # Main Loop
    # =========================

    def timer_callback(self):
        """
        Called periodically by the timer.
        Converts input data, runs AsyncVLA inference, and publishes the output.
        """

        # Prevent overlapping inference calls
        # If one inference is still running, skip this timer cycle
        if self.inference_running:
            return

        # Need at least two frames: previous and current
        if self.prev_image_msg is None or self.latest_image_msg is None:
            return

        # Mark inference as running
        self.inference_running = True

        try:
            # -----------------------------
            # Convert images
            # -----------------------------
            try:
                # Convert previous and current ROS image messages into NumPy arrays
                past_image = self.ros_image_to_numpy(self.prev_image_msg)
                current_image = self.ros_image_to_numpy(self.latest_image_msg)
            except Exception as exc:
                self.get_logger().error(f'Image conversion failed: {exc}')
                return

            # -----------------------------
            # Goal handling
            # -----------------------------
            # Default: no goal provided
            goal_text = None
            goal_pose = None
            goal_image = None

            # If a goal has been received, extract whichever fields are active
            if self.latest_goal_msg is not None:

                # Extract text goal if enabled
                if self.latest_goal_msg.use_text:
                    goal_text = self.latest_goal_msg.goal_text

                # Extract pose goal if enabled
                if self.latest_goal_msg.use_pose:
                    goal_pose = self.extract_goal_pose(self.latest_goal_msg)

                # Extract image goal if enabled
                if self.latest_goal_msg.use_image:
                    try:
                        goal_image = self.ros_image_to_numpy(
                            self.latest_goal_msg.goal_image
                        )
                    except Exception as exc:
                        self.get_logger().error(f'Goal image conversion failed: {exc}')
                        return

            # -----------------------------
            # Debug logging
            # -----------------------------
            # Print which goal modalities are currently active
            self.get_logger().info(
                f"Goal → text={goal_text is not None}, "
                f"pose={goal_pose is not None}, "
                f"image={goal_image is not None}"
            )

            # -----------------------------
            # Run AsyncVLA inference
            # -----------------------------
            try:
                # Send images and goal inputs to the model
                prediction = self.backend.predict(
                    past_image=past_image,
                    current_image=current_image,
                    goal_pose=goal_pose,
                    goal_text=goal_text,
                    goal_image=goal_image,
                )
            except Exception as exc:
                self.get_logger().error(f'AsyncVLA inference failed: {exc}')
                return

            # -----------------------------
            # Convert prediction to ROS message
            # -----------------------------
            msg = ActionChunk()

            # Add timestamp and frame info to the message header
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'

            # Convert each predicted relative pose into a Pose2D message
            poses = []
            for x, y, theta in prediction.relative_poses:
                p = Pose2D()
                p.x = float(x)
                p.y = float(y)
                p.theta = float(theta)
                poses.append(p)

            # Store the list of poses in the outgoing message
            msg.relative_poses = poses

            # -----------------------------
            # Publish result
            # -----------------------------
            self.action_pub.publish(msg)

        finally:
            # Always reset the flag, even if an error occurs
            self.inference_running = False