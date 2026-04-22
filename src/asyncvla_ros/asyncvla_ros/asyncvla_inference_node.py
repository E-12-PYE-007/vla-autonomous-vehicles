#!/usr/bin/env python3

# Enables postponed evaluation of type hints
from __future__ import annotations

# NumPy is used for image array conversion and manipulation
import numpy as np

# ROS 2 Python client library
import rclpy
from rclpy.node import Node

# ROS message types:
# - Image: incoming camera frames
# - Pose2D: individual relative poses in the predicted trajectory
# - GoalSpec: multi-modal goal input (text / pose / image)
# - ActionChunk: predicted trajectory output from AsyncVLA
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose2D
from asyncvla_interfaces.msg import GoalSpec, ActionChunk

# Backend wrapper that loads and runs AsyncVLA
from asyncvla_ros.asyncvla_backend import AsyncVLABackend


class AsyncVLAInferenceNode(Node):
    def __init__(self):
        # Initialize this ROS 2 node
        super().__init__('asyncvla_inference_node')

        # -----------------------------
        # Internal state
        # -----------------------------
        # Previous and latest camera frames are stored so the model can use
        # a short temporal history (past frame + current frame)
        self.prev_image_msg = None
        self.latest_image_msg = None

        # Store the most recent goal message received from /asyncvla/goal
        self.latest_goal_msg = None

        # Flag used to prevent overlapping inference calls
        # If one inference is still running when the timer fires again,
        # the new cycle is skipped
        self.inference_running = False

        # -----------------------------
        # Parameters
        # -----------------------------
        # Path to the AsyncVLA repository
        self.declare_parameter('model_repo_path', '')

        # Path to the trained checkpoint directory
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
        # Create the AsyncVLA backend object
        self.backend = AsyncVLABackend(
            model_repo_path=model_repo_path or None,
            checkpoint_path=checkpoint_path or None,
            device=device,
        )

        # Load the AsyncVLA model once at startup
        self.backend.load_model()

        # -----------------------------
        # Subscribers
        # -----------------------------
        # Subscribe to the camera image topic
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
        # Publish predicted action chunks (trajectory as relative poses)
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

        # Startup logs
        self.get_logger().info('AsyncVLA Inference Node started')
        self.get_logger().info(f'Inference period: {self.inference_period}s')

    # =========================
    # Callbacks
    # =========================

    def image_callback(self, msg: Image):
        """
        Store the latest two camera frames.

        The previous current frame becomes prev_image_msg,
        and the newly received frame becomes latest_image_msg.
        """
        self.prev_image_msg = self.latest_image_msg
        self.latest_image_msg = msg

    def goal_callback(self, msg: GoalSpec):
        """
        Store the most recent goal message.

        This node supports text, pose, and image goals through GoalSpec.
        """
        self.latest_goal_msg = msg

        # Print goal metadata for debugging
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

        Assumptions:
        - 3 channels
        - image is published as BGR (as done by OpenCV / cv_bridge in camera node)

        This function converts BGR -> RGB so the image is in the format expected
        by PIL / AsyncVLA downstream.
        """
        # Reject invalid image dimensions
        if msg.height == 0 or msg.width == 0:
            raise ValueError('Empty image received')

        channels = 3
        expected_size = msg.height * msg.width * channels

        # Convert raw bytes to a flat NumPy array
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        # Make sure the image buffer is large enough
        if raw.size < expected_size:
            raise ValueError(f'Image data too small: {raw.size} < {expected_size}')

        # Reshape flat array into H x W x C image
        image = raw[:expected_size].reshape((msg.height, msg.width, channels))

        # Convert BGR -> RGB because camera node publishes bgr8
        image = image[:, :, ::-1]

        return image

    def extract_goal_pose(self, msg: GoalSpec):
        """
        Extract a simple (x, y, theta) tuple from the goal message.

        At the moment theta is set to 0.0 because only position is being used
        directly from the goal pose message.
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
        Main periodic inference loop.

        Workflow:
        1. Check if inference is already running
        2. Check if at least two frames are available
        3. Convert images to NumPy arrays
        4. Extract the current goal inputs
        5. Run AsyncVLA inference through the backend
        6. Convert result into an ActionChunk ROS message
        7. Publish the result
        """

        # Prevent overlapping inference calls
        if self.inference_running:
            return

        # Need both a previous frame and a current frame
        if self.prev_image_msg is None or self.latest_image_msg is None:
            return

        # Mark inference as active
        self.inference_running = True

        try:
            # -----------------------------
            # Convert camera images
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
            # Default to no goal in any modality
            goal_text = None
            goal_pose = None
            goal_image = None

            # If a goal message has been received, extract whichever modalities are active
            if self.latest_goal_msg is not None:

                # Extract text goal
                if self.latest_goal_msg.use_text:
                    goal_text = self.latest_goal_msg.goal_text

                # Extract pose goal
                if self.latest_goal_msg.use_pose:
                    goal_pose = self.extract_goal_pose(self.latest_goal_msg)

                # Extract image goal and convert it to NumPy
                if self.latest_goal_msg.use_image:
                    try:
                        goal_image = self.ros_image_to_numpy(
                            self.latest_goal_msg.goal_image
                        )
                    except Exception as exc:
                        self.get_logger().error(f'Goal image conversion failed: {exc}')
                        return

            # Log which goal modalities are currently active
            self.get_logger().info(
                f"Goal → text={goal_text is not None}, "
                f"pose={goal_pose is not None}, "
                f"image={goal_image is not None}"
            )

            # -----------------------------
            # Run AsyncVLA inference
            # -----------------------------
            try:
                # Send images and goal information to the backend
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

            # Stamp the output message
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'

            # Convert each predicted (x, y, theta) tuple into a Pose2D
            poses = []
            for x, y, theta in prediction.relative_poses:
                p = Pose2D()
                p.x = float(x)
                p.y = float(y)
                p.theta = float(theta)
                poses.append(p)

            # Assign the trajectory to the output message
            msg.relative_poses = poses

            # -----------------------------
            # Publish result
            # -----------------------------
            self.action_pub.publish(msg)

        finally:
            # Always clear the running flag, even if an error occurs
            self.inference_running = False


# =========================
# Entry point
# =========================

def main(args=None):
    """
    Standard ROS 2 Python node entry point.
    """
    rclpy.init(args=args)
    node = AsyncVLAInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()