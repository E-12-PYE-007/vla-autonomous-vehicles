#!/usr/bin/env python3
"""Track VLA reference trajectories with encoder-odometry feedback.

Each VLA ActionChunk contains future poses relative to the robot at inference
time. This node converts that relative chunk into an odom-frame path using the
current encoder odometry pose, then repeatedly compares live odometry against
that path. The resulting cross-track and heading error corrections are converted
to left/right wheel velocity references for the wheel PID controller.
"""

import math

import rclpy
from custom_msgs.msg import ActionChunk, LeftRightFloat32
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node


WHEEL_BASE = 0.22
LOOKAHEAD_DISTANCE = 0.35
MAX_LINEAR_SPEED = 0.30
MAX_ANGULAR_SPEED = 1.20
COMMAND_TIMEOUT_SEC = 0.75
K_CROSS_TRACK = 1.6
K_HEADING = 1.0
GOAL_TOLERANCE = 0.05
CONTROL_RATE_HZ = 10.0


def wrap_to_pi(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class OdomActionChunkTrackerNode(Node):
    def __init__(self):
        super().__init__("outer_loop_controller")

        self.declare_parameter("use_sim", False)
        self.use_sim = bool(self.get_parameter("use_sim").value)

        self.current_pose = None
        self.reference_path = []
        self.last_action_time = None
        self.closest_idx = 0
        self.seq_num = 1

        self.create_subscription(
            Pose2D,
            "odom_pose2d",
            self.pose_callback,
            10,
        )
        self.create_subscription(
            ActionChunk,
            "/asyncvla/action_chunk",
            self.action_callback,
            10,
        )
        self.create_subscription(
            ActionChunk,
            "/ticvla/action_chunk",
            self.action_callback,
            10,
        )

        if self.use_sim:
            self.publisher = self.create_publisher(
                Twist,
                "cmd_vel",
                10,
            )
        else:
            self.publisher = self.create_publisher(
                LeftRightFloat32,
                "wheel_velocity_reference",
                10,
            )

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)

        self.get_logger().info("Odometry-aware ActionChunk tracker started")

    @staticmethod
    def clamp(value, low, high):
        """Clamp a controller command to robot limits."""
        return max(low, min(high, value))

    def pose_callback(self, msg):
        """Store the latest encoder-odometry pose estimate."""
        self.current_pose = msg

    def action_callback(self, msg):
        """Anchor a new robot-relative VLA chunk in the current odom frame."""
        if self.current_pose is None:
            self.get_logger().warn("Ignoring ActionChunk until odometry pose is available")
            return
        if not msg.relative_poses:
            self.get_logger().warn("Ignoring empty ActionChunk")
            return

        anchor = self.current_pose
        cos_yaw = math.cos(anchor.theta)
        sin_yaw = math.sin(anchor.theta)
        path = []

        for relative in msg.relative_poses:
            rel_x = float(relative.x)
            rel_y = float(relative.y)
            odom_x = anchor.x + cos_yaw * rel_x - sin_yaw * rel_y
            odom_y = anchor.y + sin_yaw * rel_x + cos_yaw * rel_y
            odom_yaw = wrap_to_pi(anchor.theta + float(relative.theta))
            path.append((odom_x, odom_y, odom_yaw))

        self.reference_path = path
        self.closest_idx = 0
        self.last_action_time = self.get_clock().now()

    def find_tracking_target(self):
        """Find closest path point and a lookahead target ahead of it."""
        if self.current_pose is None or not self.reference_path:
            return None, None

        x = self.current_pose.x
        y = self.current_pose.y
        search_start = max(0, self.closest_idx - 2)
        search_end = len(self.reference_path)

        best_idx = self.closest_idx
        best_dist = float("inf")
        for idx in range(search_start, search_end):
            px, py, _ = self.reference_path[idx]
            dist = math.hypot(px - x, py - y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        self.closest_idx = best_idx

        target_idx = len(self.reference_path) - 1
        for idx in range(best_idx, len(self.reference_path)):
            px, py, _ = self.reference_path[idx]
            if math.hypot(px - x, py - y) >= LOOKAHEAD_DISTANCE:
                target_idx = idx
                break

        return self.reference_path[best_idx], self.reference_path[target_idx]

    def control_loop(self):
        """Compute path-tracking error from odometry and publish wheel refs."""
        if self.current_pose is None or not self.reference_path or self.last_action_time is None:
            self.publish_wheel_refs(0.0, 0.0)
            return

        age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
        if age > COMMAND_TIMEOUT_SEC:
            self.publish_wheel_refs(0.0, 0.0)
            return

        closest, target = self.find_tracking_target()
        if closest is None or target is None:
            self.publish_wheel_refs(0.0, 0.0)
            return

        x = self.current_pose.x
        y = self.current_pose.y
        yaw = self.current_pose.theta
        tx, ty, tyaw = target

        dx = tx - x
        dy = ty - y
        distance_to_target = math.hypot(dx, dy)
        if distance_to_target < GOAL_TOLERANCE and target == self.reference_path[-1]:
            self.publish_wheel_refs(0.0, 0.0)
            return

        # Express target error in the robot frame. local_x is forward error;
        # local_y is lateral error. This is the odometry feedback term the
        # trajectory controller uses to correct path tracking.
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading_error = wrap_to_pi(tyaw - yaw)

        linear = self.clamp(local_x, 0.0, MAX_LINEAR_SPEED)
        angular = K_CROSS_TRACK * local_y + K_HEADING * heading_error
        angular = self.clamp(angular, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        left = linear - 0.5 * WHEEL_BASE * angular
        right = linear + 0.5 * WHEEL_BASE * angular
        self.publish_wheel_refs(left, right)

    def publish_wheel_refs(self, left, right):
        """Publish desired wheel speeds — Twist for sim, LeftRightFloat32 for hardware."""
        if self.use_sim:
            msg = Twist()
            msg.linear.x = (left + right) / 2.0
            msg.angular.z = (right - left) / WHEEL_BASE
            self.publisher.publish(msg)
        else:
            msg = LeftRightFloat32()
            msg.left = float(left)
            msg.right = float(right)
            msg.seq_num = self.seq_num
            self.publisher.publish(msg)
            self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = OdomActionChunkTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_wheel_refs(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
