#!/usr/bin/env python3
"""Faithful port of the TIC-VLA benchmark waypoint controller.

Source: DynaNav/behavior/nova_carter_test_ticvla.py (upstream's deployed benchmark
driver). Key properties, kept as-is:

- Consumes the waypoint chunk open-loop in the robot frame (no odometry anchoring;
  the chunk refreshes at 10 Hz so staleness stays under ~0.1 s).
- Lookahead by ARC LENGTH along the waypoint polyline (not straight-line distance),
  with the index clamped to [2, T-3].
- Speed law v = min(v_max, w_max/|kappa|): drive at v_max, slow only for sharp turns.
  Speed never scales with waypoint magnitude, so timid model output cannot stall the
  robot into the vx=0 regime it was never trained on.
- Steering = reduced curvature feedforward + low-pass-filtered bearing feedback.
  Per-waypoint theta is never used (the model does not produce headings).
- Slew-rate limiting on both commands.

Deviation from upstream: V_MAX is 0.30 m/s (our platform limit) instead of 1.5.
Upstream's stuck-detection/backup recovery behaviour is not ported.
"""

import math

import numpy as np
import rclpy
from custom_msgs.msg import ActionChunk, LeftRightFloat32
from geometry_msgs.msg import PointStamped, Twist
from rclpy.node import Node


WHEEL_BASE = 0.22
CONTROL_RATE_HZ = 10.0
COMMAND_TIMEOUT_SEC = 0.75

# --- Values from nova_carter_test_ticvla.py ---
# Upstream uses 1.5 m/s (a simulated Nova Carter in Isaac Sim). 0.30 is this platform's
# real limit, set by the wheel joint velocity cap in robot.xacro (10 rad/s * 0.035 m
# = 0.35 m/s). Kept at the platform limit so sim behaviour predicts hardware.
V_MAX = 0.30              # m/s
W_MAX = 1.20              # rad/s (upstream value)
L_DES = 1.0               # m — lookahead arc length along the polyline
K_ANGULAR = 0.8           # bearing feedback gain
FF_SCALE = 0.5            # curvature feedforward scale ("reduced feedforward")
ALPHA_FILTER = 0.35       # yaw-error low-pass smoothing
EPS = 1e-3

# Not in upstream. A chunk is a PLAN spanning PLAN_HORIZON_SEC, but pure pursuit executes
# its implied heading change as fast as the robot physically can. On short plans that is a
# problem: kappa = 2y/L^2 explodes as L shrinks, and the bearing term is large for any
# sideways point regardless of distance, so w saturates and the robot turns ~3x further
# than the plan asked for before the next chunk lands. Capping w at (heading change) /
# (plan duration) makes the turn take as long as the plan says it should. Self-scaling:
# no effect on straight plans, strong effect on short lateral ones.
PLAN_HORIZON_SEC = 3.0    # ACTION_HORIZON_STEPS(30) / 10 Hz

MAX_ACCEL_LIN = 2.0       # m/s^2
MAX_DECEL_LIN = 2.5       # m/s^2
MAX_ACCEL_ANG = 3.0       # rad/s^2
MAX_DECEL_ANG = 3.5       # rad/s^2
LIN_DEADBAND = 0.001      # m/s
ANG_DEADBAND = 0.0005     # rad/s


def slew(cur, tgt, accel, decel, dt, deadband):
    """Rate-limit cur toward tgt; verbatim port of upstream _slew."""
    dt = max(1e-4, float(dt))
    dv = tgt - cur
    limit = accel if dv > 0.0 else decel
    max_step = limit * dt
    if dv > max_step:
        cur += max_step
    elif dv < -max_step:
        cur -= max_step
    else:
        cur = tgt
    if abs(cur) < deadband and abs(tgt) < deadband:
        cur = 0.0
    return cur


class TicControllerNode(Node):
    def __init__(self):
        super().__init__("tic_controller")

        self.declare_parameter("use_sim", False)
        self.use_sim = bool(self.get_parameter("use_sim").value)

        self.last_action = None
        self.last_action_time = None
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.yaw_err_filt = None
        self.seq_num = 1

        self.create_subscription(ActionChunk, "/asyncvla/action_chunk", self.action_callback, 10)
        self.create_subscription(ActionChunk, "/ticvla/action_chunk", self.action_callback, 10)

        if self.use_sim:
            self.publisher = self.create_publisher(Twist, "cmd_vel", 10)
        else:
            self.publisher = self.create_publisher(LeftRightFloat32, "wheel_velocity_reference", 10)

        # Debug/analysis: the lookahead point actually chosen each cycle (base_link frame)
        self.lookahead_pub = self.create_publisher(PointStamped, "tic_controller/lookahead", 10)

        self.dt = 1.0 / CONTROL_RATE_HZ
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("TIC benchmark controller started (arc-length pure pursuit)")

    def action_callback(self, msg):
        if not msg.relative_poses:
            self.get_logger().warn("Ignoring empty action chunk")
            return
        self.last_action = msg
        self.last_action_time = self.get_clock().now()

    def compute_target(self):
        """Waypoints -> (target_v, target_w); port of the upstream conversion block."""
        wps = np.array(
            [[p.x, p.y] for p in self.last_action.relative_poses], dtype=np.float64
        )  # (T, 2), cumulative offsets in the robot frame
        T = len(wps)
        if T < 2:
            return 0.0, 0.0

        # Arc length along the polyline
        inc = np.diff(wps, axis=0)
        seg = np.hypot(inc[:, 0], inc[:, 1])
        s = np.concatenate([[0.0], np.cumsum(seg)])

        # First index whose arc length reaches L_DES, clamped to [2, T-3].
        # With short chunks this lands near the end of the path — same behaviour as
        # upstream, which runs the identical clamp on T=30 chunks.
        j = int(np.searchsorted(s, L_DES, side="left"))
        j = int(np.clip(j, min(2, T - 1), max(T - 3, 0)))

        xL, yL = float(wps[j, 0]), float(wps[j, 1])
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "base_link"
        pt.point.x, pt.point.y = xL, yL
        self.lookahead_pub.publish(pt)

        L = float(np.hypot(xL, yL))
        if L < EPS:
            return 0.0, 0.0

        # Low-pass-filtered bearing to the lookahead point
        yaw_err = math.atan2(yL, xL)
        if self.yaw_err_filt is None:
            self.yaw_err_filt = yaw_err
        e = math.atan2(math.sin(yaw_err - self.yaw_err_filt),
                       math.cos(yaw_err - self.yaw_err_filt))
        self.yaw_err_filt += ALPHA_FILTER * e

        # Pure pursuit curvature; speed limited by turn sharpness, not waypoint size
        kappa = 2.0 * yL / (L * L)
        v_kappa = W_MAX / (abs(kappa) + EPS)
        v_cmd = float(np.clip(min(V_MAX, v_kappa), 0.0, V_MAX))

        w_ff = FF_SCALE * v_cmd * kappa
        w_fb = K_ANGULAR * self.yaw_err_filt
        # Take no longer than W_MAX allows, but also no faster than the plan implies.
        w_limit = min(W_MAX, abs(self.yaw_err_filt) / PLAN_HORIZON_SEC)
        w_cmd = float(np.clip(w_ff + w_fb, -w_limit, w_limit))
        return v_cmd, w_cmd

    def control_loop(self):
        if self.last_action is None or self.last_action_time is None:
            target_v, target_w = 0.0, 0.0
        else:
            age = (self.get_clock().now() - self.last_action_time).nanoseconds / 1e9
            if age > COMMAND_TIMEOUT_SEC:
                target_v, target_w = 0.0, 0.0
            else:
                target_v, target_w = self.compute_target()

        self.cmd_v = slew(self.cmd_v, target_v, MAX_ACCEL_LIN, MAX_DECEL_LIN, self.dt, LIN_DEADBAND)
        self.cmd_w = slew(self.cmd_w, target_w, MAX_ACCEL_ANG, MAX_DECEL_ANG, self.dt, ANG_DEADBAND)
        self.publish_command(self.cmd_v, self.cmd_w)

        # The sys1 diagnostic logs the MEASURED angular rate from /odom, which includes
        # the sim's drive dynamics and is not what this controller asked for. Log the
        # command itself so the two can be told apart.
        self._log_n = getattr(self, "_log_n", 0) + 1
        if self._log_n % 10 == 0:  # 1 Hz at CONTROL_RATE_HZ
            span = 0.0
            bearing = 0.0
            if self.last_action is not None and self.last_action.relative_poses:
                p = self.last_action.relative_poses[-1]
                span = math.hypot(p.x, p.y)
                bearing = math.degrees(math.atan2(p.y, p.x))
            cap = min(W_MAX, abs(self.yaw_err_filt) / PLAN_HORIZON_SEC) if self.yaw_err_filt is not None else W_MAX
            self.get_logger().info(
                f"cmd v={self.cmd_v:+.3f} w={self.cmd_w:+.3f} | "
                f"target v={target_v:+.3f} w={target_w:+.3f} | "
                f"w_cap={cap:+.3f} | chunk span={span:.2f}m bearing={bearing:+.0f}deg"
            )

    def publish_command(self, v, w):
        if self.use_sim:
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(w)
            self.publisher.publish(msg)
        else:
            msg = LeftRightFloat32()
            msg.left = float(v - 0.5 * WHEEL_BASE * w)
            msg.right = float(v + 0.5 * WHEEL_BASE * w)
            msg.seq_num = self.seq_num
            self.publisher.publish(msg)
            self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = TicControllerNode()
    rclpy.spin(node)
    node.publish_command(0.0, 0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
