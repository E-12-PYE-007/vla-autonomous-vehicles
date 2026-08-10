#!/usr/bin/env python3
"""PD waypoint controller for AsyncVLA, matching the reference implementation.

Follows AsyncVLA/inference/run_asyncvla.py `pd_controller`: select waypoint
index 4 (midpoint of an 8-step chunk), drive toward it with a linear/angular
"reach in DT seconds" law, then apply turn-shape-preserving velocity limits.

The reference computes PD once per chunk in the chunk's own robot frame. To
publish at 10 Hz between chunks, we anchor the chosen waypoint in the odom
frame when a chunk arrives, then transform it back into the current robot
frame each tick so live odometry closes the loop.
"""

import math

import rclpy
from custom_msgs.msg import ActionChunk, LeftRightFloat32
from geometry_msgs.msg import Pose2D, Twist
from rclpy.node import Node


WAYPOINT_SELECT = 4          # midpoint of the 8-step chunk (matches reference)
DT = 1.0 / 3.0               # PD "reach in DT seconds" horizon (matches reference)
CONTROL_RATE_HZ = 10.0
COMMAND_TIMEOUT_SEC = 0.75

# Velocity limits from reference pd_controller
V_HARD_MAX = 0.5             # pre-shape clip on linear vel
W_HARD_MAX = 1.0             # pre-shape clip on angular vel
MAXV = 0.3                   # turn-shape-preserving max linear
MAXW = 0.3                   # turn-shape-preserving max angular

WHEEL_BASE = 0.22
EPS = 1e-8


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def sign(x):
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


class AsyncPDControllerNode(Node):
    def __init__(self):
        super().__init__("async_pd_controller")

        self.declare_parameter("use_sim", False)
        self.use_sim = bool(self.get_parameter("use_sim").value)

        self.current_pose = None
        self.target_x = None
        self.target_y = None
        self.target_theta = None
        self.last_action_time = None
        self.seq_num = 1

        self.create_subscription(Pose2D, "odom_pose2d", self.pose_callback, 10)
        self.create_subscription(
            ActionChunk, "/asyncvla/action_chunk", self.action_callback, 10
        )

        if self.use_sim:
            self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        else:
            self.publisher = self.create_publisher(
                LeftRightFloat32, "wheel_velocity_reference", 10
            )

        self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.get_logger().info("Async PD waypoint controller started")

    def pose_callback(self, msg):
        self.current_pose = msg

    def action_callback(self, msg):
        """Anchor waypoint index 4 of the incoming chunk in the odom frame."""
        if self.current_pose is None:
            self.get_logger().warn("Ignoring ActionChunk until odometry is available")
            return
        if len(msg.relative_poses) <= WAYPOINT_SELECT:
            self.get_logger().warn(
                f"Chunk has {len(msg.relative_poses)} waypoints, need > {WAYPOINT_SELECT}"
            )
            return

        wp = msg.relative_poses[WAYPOINT_SELECT]
        anchor = self.current_pose
        cos_yaw = math.cos(anchor.theta)
        sin_yaw = math.sin(anchor.theta)

        self.target_x = anchor.x + cos_yaw * float(wp.x) - sin_yaw * float(wp.y)
        self.target_y = anchor.y + sin_yaw * float(wp.x) + cos_yaw * float(wp.y)
        self.target_theta = wrap_to_pi(anchor.theta + float(wp.theta))
        self.last_action_time = self.get_clock().now()

    def control_loop(self):
        if (
            self.current_pose is None
            or self.target_x is None
            or self.last_action_time is None
        ):
            self.publish_cmd(0.0, 0.0)
            return

        age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
        if age > COMMAND_TIMEOUT_SEC:
            self.publish_cmd(0.0, 0.0)
            return

        # Express anchored target in the current robot frame.
        dx_w = self.target_x - self.current_pose.x
        dy_w = self.target_y - self.current_pose.y
        yaw = self.current_pose.theta
        dx = math.cos(yaw) * dx_w + math.sin(yaw) * dy_w
        dy = -math.sin(yaw) * dx_w + math.cos(yaw) * dy_w
        dtheta = wrap_to_pi(self.target_theta - yaw)
        hx = math.cos(dtheta)
        hy = math.sin(dtheta)

        v, omega = self._pd(dx, dy, hx, hy)
        self.publish_cmd(v, omega)

    @staticmethod
    def _pd(dx, dy, hx, hy):
        """Reference PD law from AsyncVLA/inference/run_asyncvla.py."""
        if abs(dx) < EPS and abs(dy) < EPS:
            v = 0.0
            omega = wrap_to_pi(math.atan2(hy, hx)) / DT
        elif abs(dx) < EPS:
            v = 0.0
            omega = sign(dy) * math.pi / (2.0 * DT)
        else:
            v = dx / DT
            omega = math.atan(dy / dx) / DT

        # Hard clip
        v = max(0.0, min(V_HARD_MAX, v))
        omega = max(-W_HARD_MAX, min(W_HARD_MAX, omega))

        return AsyncPDControllerNode._limit(v, omega)

    @staticmethod
    def _limit(v, w):
        """Turn-shape-preserving velocity limiter from the reference."""
        if abs(v) <= MAXV:
            if abs(w) <= MAXW:
                return v, w
            rd = v / w
            return MAXW * sign(v) * abs(rd), MAXW * sign(w)
        if abs(w) <= 0.001:
            return MAXV * sign(v), 0.0
        rd = v / w
        if abs(rd) >= MAXV / MAXW:
            return MAXV * sign(v), MAXV * sign(w) / abs(rd)
        return MAXW * sign(v) * abs(rd), MAXW * sign(w)

    def publish_cmd(self, v, omega):
        """Twist for sim, left/right wheel refs for hardware."""
        if self.use_sim:
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(omega)
            self.publisher.publish(msg)
        else:
            left = v - 0.5 * WHEEL_BASE * omega
            right = v + 0.5 * WHEEL_BASE * omega
            msg = LeftRightFloat32()
            msg.left = float(left)
            msg.right = float(right)
            msg.seq_num = self.seq_num
            self.publisher.publish(msg)
            self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = AsyncPDControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
