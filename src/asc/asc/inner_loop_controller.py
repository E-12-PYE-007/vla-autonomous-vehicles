#!/usr/bin/env python3
"""Closed-loop wheel velocity PID controller for the ASClinic robot.

Inputs:
- `wheel_velocity_reference`: desired left/right wheel speeds in m/s
- `encoder_counts`: Roboclaw encoder delta counts per control interval

Output:
- `set_motor_duty_cycle`: left/right motor duty cycle percentages for Roboclaw
"""

import numpy as np
import rclpy
from custom_msgs.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node


CONTROL_PERIOD_SEC = 0.1
WHEEL_RADIUS = 0.072
ENCODER_COUNTS_PER_REV = 4480
KP_LEFT = 10.0
KI_LEFT = 5.0
KD_LEFT = 2.0
KP_RIGHT = 10.0
KI_RIGHT = 5.0
KD_RIGHT = 2.0
REF_TIMEOUT_SEC = 0.5
ENCODER_TIMEOUT_SEC = 0.25
WHEEL_SPEED_FILTER_ALPHA = 0.5
MOTOR_FILTER_ALPHA = 0.6


class WheelPIDControllerNode(Node):
    def __init__(self):
        super().__init__("inner_loop_controller")

        self.v_left_ref = 0.0
        self.v_right_ref = 0.0
        self.v_left_meas = 0.0
        self.v_right_meas = 0.0
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.prev_left_error = 0.0
        self.prev_right_error = 0.0
        self.prev_duty_left = 0.0
        self.prev_duty_right = 0.0
        self.last_ref_time = self.get_clock().now()
        self.last_encoder_time = None
        self.seq_num = 1

        self.create_subscription(
            LeftRightFloat32,
            "wheel_velocity_reference",
            self.reference_callback,
            10,
        )
        self.create_subscription(
            LeftRightInt32,
            "encoder_counts",
            self.encoder_callback,
            10,
        )
        self.publisher = self.create_publisher(
            LeftRightFloat32,
            "set_motor_duty_cycle",
            10,
        )
        self.timer = self.create_timer(CONTROL_PERIOD_SEC, self.control_loop)

    def reference_callback(self, msg):
        """Update desired wheel speeds and refresh the command watchdog."""
        self.v_left_ref = float(msg.left)
        self.v_right_ref = float(msg.right)
        self.last_ref_time = self.get_clock().now()

    def encoder_callback(self, msg):
        """Convert encoder delta counts into filtered wheel speed measurements."""
        now = self.get_clock().now()
        encoder_dt = CONTROL_PERIOD_SEC
        if self.last_encoder_time is not None:
            encoder_dt = max((now - self.last_encoder_time).nanoseconds / 1e9, 1e-6)
        self.last_encoder_time = now

        distance_per_count = (2.0 * np.pi * WHEEL_RADIUS) / ENCODER_COUNTS_PER_REV
        raw_left = (float(msg.left) * distance_per_count) / encoder_dt
        raw_right = (float(msg.right) * distance_per_count) / encoder_dt
        self.v_left_meas = WHEEL_SPEED_FILTER_ALPHA * raw_left + (1.0 - WHEEL_SPEED_FILTER_ALPHA) * self.v_left_meas
        self.v_right_meas = WHEEL_SPEED_FILTER_ALPHA * raw_right + (1.0 - WHEEL_SPEED_FILTER_ALPHA) * self.v_right_meas

    @staticmethod
    def feedforward_left(v):
        """Left wheel feedforward duty estimate from ASClinic motor tuning."""
        if abs(v) < 0.02:
            return 0.0
        return (v + 0.054326) / 0.013121

    @staticmethod
    def feedforward_right(v):
        """Right wheel feedforward duty estimate from ASClinic motor tuning."""
        if abs(v) < 0.02:
            return 0.0
        return (v + 0.039647) / 0.012822

    def control_loop(self):
        """Run feedforward + PID and publish bounded/smoothed duty cycles."""
        age = (self.get_clock().now() - self.last_ref_time).nanoseconds / 1e9
        if age > REF_TIMEOUT_SEC:
            # If high-level commands stop, clear references and integral memory so
            # the robot stops instead of continuing on stale accumulated error.
            self.v_left_ref = 0.0
            self.v_right_ref = 0.0
            self.integral_left = 0.0
            self.integral_right = 0.0

        encoder_stale = self.last_encoder_time is None or (self.get_clock().now() - self.last_encoder_time).nanoseconds / 1e9 > ENCODER_TIMEOUT_SEC
        if encoder_stale:
            self.v_left_ref = 0.0
            self.v_right_ref = 0.0
            self.integral_left = 0.0
            self.integral_right = 0.0

        left_error = self.v_left_ref - self.v_left_meas
        right_error = self.v_right_ref - self.v_right_meas
        d_left = (left_error - self.prev_left_error) / CONTROL_PERIOD_SEC
        d_right = (right_error - self.prev_right_error) / CONTROL_PERIOD_SEC

        self.integral_left = float(np.clip(self.integral_left + left_error * CONTROL_PERIOD_SEC, -50.0, 50.0))
        self.integral_right = float(np.clip(self.integral_right + right_error * CONTROL_PERIOD_SEC, -50.0, 50.0))

        duty_left = 0.97 * self.feedforward_left(self.v_left_ref) + KP_LEFT * left_error + KI_LEFT * self.integral_left + KD_LEFT * d_left
        duty_right = 0.99 * self.feedforward_right(self.v_right_ref) + KP_RIGHT * right_error + KI_RIGHT * self.integral_right + KD_RIGHT * d_right

        self.prev_left_error = left_error
        self.prev_right_error = right_error

        # Bound to Roboclaw percent duty convention and smooth motor commands to
        # reduce sharp current spikes from sudden model output changes.
        duty_left = float(np.clip(duty_left, -100.0, 100.0))
        duty_right = float(np.clip(duty_right, -100.0, 100.0))
        duty_left = MOTOR_FILTER_ALPHA * duty_left + (1.0 - MOTOR_FILTER_ALPHA) * self.prev_duty_left
        duty_right = MOTOR_FILTER_ALPHA * duty_right + (1.0 - MOTOR_FILTER_ALPHA) * self.prev_duty_right
        self.prev_duty_left = duty_left
        self.prev_duty_right = duty_right
        self.publish_duty(duty_left, duty_right)

    def publish_duty(self, left, right):
        """Publish duty cycle percentages for the Roboclaw driver node."""
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.publisher.publish(msg)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = WheelPIDControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.publish_duty(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
