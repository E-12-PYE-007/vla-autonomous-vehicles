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
from asclinic_vla_interfaces.msg import ActionChunk, LeftRightFloat32
from geometry_msgs.msg import Pose2D
from rclpy.node import Node


def wrap_to_pi(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class OdomActionChunkTrackerNode(Node):
    def __init__(self):
        super().__init__('odom_action_chunk_tracker')

        self.declare_parameter('action_topic', '/asyncvla/action_chunk')
        self.declare_parameter('pose2d_topic', 'odom_pose2d')
        self.declare_parameter('wheel_reference_topic', 'wheel_velocity_reference')
        self.declare_parameter('wheel_base', 0.22)
        self.declare_parameter('lookahead_distance', 0.35)
        self.declare_parameter('max_linear_speed', 0.30)
        self.declare_parameter('max_angular_speed', 1.20)
        self.declare_parameter('command_timeout_sec', 0.75)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('k_cross_track', 1.6)
        self.declare_parameter('k_heading', 1.0)
        self.declare_parameter('goal_tolerance', 0.05)

        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.lookahead_distance = float(self.get_parameter('lookahead_distance').value)
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.command_timeout_sec = float(self.get_parameter('command_timeout_sec').value)
        self.k_cross_track = float(self.get_parameter('k_cross_track').value)
        self.k_heading = float(self.get_parameter('k_heading').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)

        self.current_pose = None
        self.reference_path = []
        self.last_action_time = None
        self.closest_idx = 0
        self.seq_num = 1

        self.create_subscription(
            Pose2D,
            self.get_parameter('pose2d_topic').value,
            self.pose_callback,
            10,
        )
        self.create_subscription(
            ActionChunk,
            self.get_parameter('action_topic').value,
            self.action_callback,
            10,
        )
        self.publisher = self.create_publisher(
            LeftRightFloat32,
            self.get_parameter('wheel_reference_topic').value,
            10,
        )

        control_rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(control_rate_hz, 1.0), self.control_loop)

        self.get_logger().info('Odometry-aware ActionChunk tracker started')

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
            self.get_logger().warn('Ignoring ActionChunk until odometry pose is available')
            return
        if not msg.relative_poses:
            self.get_logger().warn('Ignoring empty ActionChunk')
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
        best_dist = float('inf')
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
            if math.hypot(px - x, py - y) >= self.lookahead_distance:
                target_idx = idx
                break

        return self.reference_path[best_idx], self.reference_path[target_idx]

    def control_loop(self):
        """Compute path-tracking error from odometry and publish wheel refs."""
        if self.current_pose is None or not self.reference_path or self.last_action_time is None:
            self.publish_wheel_refs(0.0, 0.0)
            return

        age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
        if age > self.command_timeout_sec:
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
        if distance_to_target < self.goal_tolerance and target == self.reference_path[-1]:
            self.publish_wheel_refs(0.0, 0.0)
            return

        # Express target error in the robot frame. local_x is forward error;
        # local_y is lateral error. This is the odometry feedback term the
        # trajectory controller uses to correct path tracking.
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading_error = wrap_to_pi(tyaw - yaw)

        linear = self.clamp(local_x, 0.0, self.max_linear_speed)
        angular = self.k_cross_track * local_y + self.k_heading * heading_error
        angular = self.clamp(angular, -self.max_angular_speed, self.max_angular_speed)

        left = linear - 0.5 * self.wheel_base * angular
        right = linear + 0.5 * self.wheel_base * angular
        self.publish_wheel_refs(left, right)

    def publish_wheel_refs(self, left, right):
        """Publish desired wheel speeds for the low-level PID controller."""
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


if __name__ == '__main__':
    main()
