#!/usr/bin/env python3

# Import the math library for calculations such as distance, atan2, and angle wrapping
import math

# Import ROS 2 Python client library
import rclpy
from rclpy.node import Node

# Import Twist message for velocity commands
from geometry_msgs.msg import Twist

# Import custom ActionChunk message containing predicted relative poses
from asyncvla_interfaces.msg import ActionChunk


class AsyncVLAControllerNode(Node):
    def __init__(self):
        # Initialize the ROS 2 node with the name 'asyncvla_controller_node'
        super().__init__('asyncvla_controller_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        # Topic to subscribe to for incoming predicted action chunks
        self.declare_parameter('action_topic', '/asyncvla/action_chunk')

        # Topic to publish velocity commands to
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # Index of the pose in the predicted trajectory to use as the control target
        # Using a later point gives lookahead behavior instead of reacting only to the first point
        self.declare_parameter('lookahead_index', 1)

        # Proportional gain for linear velocity
        self.declare_parameter('k_linear', 0.8)

        # Proportional gain for angular velocity
        self.declare_parameter('k_angular', 1.5)

        # Maximum allowed forward/backward speed
        self.declare_parameter('max_linear_speed', 0.35)

        # Maximum allowed angular speed
        self.declare_parameter('max_angular_speed', 1.2)

        # Smoothing factor for commands
        # Higher alpha = respond faster, lower alpha = smoother but slower response
        self.declare_parameter('smoothing_alpha', 0.35)

        # Time in seconds after which old action chunks are considered stale
        self.declare_parameter('command_timeout_sec', 0.75)

        # Control loop frequency in Hz
        self.declare_parameter('control_rate_hz', 10.0)

        # Minimum forward speed to help avoid stalling
        self.declare_parameter('min_linear_speed', 0.0)

        # Read topic parameters
        action_topic = self.get_parameter('action_topic').value
        cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        # Read controller parameters
        self.lookahead_index = int(self.get_parameter('lookahead_index').value)
        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)

        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)

        self.smoothing_alpha = float(self.get_parameter('smoothing_alpha').value)
        self.command_timeout_sec = float(self.get_parameter('command_timeout_sec').value)
        self.control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.min_linear_speed = float(self.get_parameter('min_linear_speed').value)

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        # Subscribe to action chunks predicted by the AsyncVLA inference node
        self.subscription = self.create_subscription(
            ActionChunk,
            action_topic,
            self.action_callback,
            10
        )

        # Publisher for sending velocity commands to the robot
        self.publisher = self.create_publisher(
            Twist,
            cmd_vel_topic,
            10
        )

        # -----------------------------
        # Internal state
        # -----------------------------
        # Store the latest received action chunk and the time it was received
        self.last_action_chunk = None
        self.last_action_time = None

        # Store previous commanded velocities for smoothing
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0

        # Create a timer to run the control loop at the desired frequency
        period = 1.0 / self.control_rate_hz
        self.timer = self.create_timer(period, self.control_loop)

        # Log startup information
        self.get_logger().info('AsyncVLA Controller Node started')
        self.get_logger().info(
            f'Listening on {action_topic}, publishing on {cmd_vel_topic}'
        )

    # ---------------------------------
    # Helper functions
    # ---------------------------------
    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """
        Restrict a value so it stays between min_value and max_value.
        """
        return max(min(value, max_value), min_value)

    @staticmethod
    def wrap_angle(angle: float) -> float:
        """
        Wrap an angle into the range [-pi, pi].
        This avoids angle discontinuities when turning.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def smooth(self, previous: float, new_value: float) -> float:
        """
        Apply exponential smoothing to reduce sudden jumps in commands.
        """
        alpha = self.smoothing_alpha
        return (1.0 - alpha) * previous + alpha * new_value

    # ---------------------------------
    # Callback function
    # ---------------------------------
    def action_callback(self, msg: ActionChunk):
        """
        Called whenever a new ActionChunk message is received.
        Saves the latest trajectory and timestamp.
        """
        # Ignore empty trajectories
        if len(msg.relative_poses) == 0:
            self.get_logger().warn('Received empty ActionChunk')
            return

        # Store the latest trajectory and the current time
        self.last_action_chunk = msg
        self.last_action_time = self.get_clock().now()

    # ---------------------------------
    # Main control loop
    # ---------------------------------
    def control_loop(self):
        """
        Runs periodically.
        Converts the latest predicted trajectory into a cmd_vel command.
        """
        # Create an empty Twist message
        cmd = Twist()

        # Stop the robot if no trajectory has been received yet
        if self.last_action_chunk is None or self.last_action_time is None:
            self.publisher.publish(cmd)
            return

        # Compute how old the last received action chunk is
        age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9

        # If the trajectory is too old, stop the robot
        if age > self.command_timeout_sec:
            self.last_cmd_linear = 0.0
            self.last_cmd_angular = 0.0
            self.publisher.publish(cmd)
            self.get_logger().warn('Trajectory timed out, publishing zero cmd_vel')
            return

        # Get the predicted relative poses from the latest action chunk
        poses = self.last_action_chunk.relative_poses

        # Choose a lookahead pose instead of always using the first point
        idx = min(self.lookahead_index, len(poses) - 1)
        target = poses[idx]

        # Extract relative target pose in the robot frame
        x = float(target.x)
        y = float(target.y)
        theta = float(target.theta)

        # Compute distance to the target point
        distance = math.hypot(x, y)

        # Compute heading error: angle from robot forward direction to target point
        heading_error = math.atan2(y, x)

        # Use the target pose orientation as an additional heading signal
        theta_error = self.wrap_angle(theta)

        # -----------------------------
        # Raw proportional control
        # -----------------------------
        # Linear speed increases with distance to the target
        linear_cmd = self.k_linear * distance

        # Angular speed depends on heading error and target orientation error
        angular_cmd = self.k_angular * (heading_error + 0.5 * theta_error)

        # -----------------------------
        # Forward motion rules
        # -----------------------------
        # If the target is behind the robot, do not drive backward
        if x < 0.0:
            linear_cmd = 0.0

        # Enforce a minimum forward speed when moving ahead
        if linear_cmd > 0.0:
            linear_cmd = max(linear_cmd, self.min_linear_speed)

        # -----------------------------
        # Limit command magnitudes
        # -----------------------------
        linear_cmd = self.clamp(linear_cmd, -self.max_linear_speed, self.max_linear_speed)
        angular_cmd = self.clamp(angular_cmd, -self.max_angular_speed, self.max_angular_speed)

        # -----------------------------
        # Smooth commands
        # -----------------------------
        linear_cmd = self.smooth(self.last_cmd_linear, linear_cmd)
        angular_cmd = self.smooth(self.last_cmd_angular, angular_cmd)

        # Save smoothed commands for use in the next cycle
        self.last_cmd_linear = linear_cmd
        self.last_cmd_angular = angular_cmd

        # Fill the Twist message
        cmd.linear.x = linear_cmd
        cmd.angular.z = angular_cmd

        # Publish velocity command
        self.publisher.publish(cmd)

        # Print debug info
        self.get_logger().info(
            f'target_idx={idx} x={x:.2f} y={y:.2f} theta={theta:.2f} '
            f'-> v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}'
        )


def main(args=None):
    """
    Main entry point for the node.
    Initializes ROS 2, creates the node, and keeps it running.
    """
    rclpy.init(args=args)
    node = AsyncVLAControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()