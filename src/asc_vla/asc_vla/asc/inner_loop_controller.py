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
from asclinic_vla_interfaces.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node


class WheelPIDControllerNode(Node):
    def __init__(self):
        super().__init__('wheel_pid_controller')

        # These parameters mirror the ASClinic tuning values, but are surfaced in
        # YAML so they can be adjusted on the robot without editing Python code.
        self.declare_parameter('wheel_reference_topic', 'wheel_velocity_reference')
        self.declare_parameter('encoder_counts_topic', 'encoder_counts')
        self.declare_parameter('duty_cycle_topic', 'set_motor_duty_cycle')
        self.declare_parameter('control_period_sec', 0.1)
        self.declare_parameter('wheel_radius', 0.072)
        self.declare_parameter('encoder_counts_per_rev', 4480)
        self.declare_parameter('kp_left', 10.0)
        self.declare_parameter('ki_left', 5.0)
        self.declare_parameter('kd_left', 2.0)
        self.declare_parameter('kp_right', 10.0)
        self.declare_parameter('ki_right', 5.0)
        self.declare_parameter('kd_right', 2.0)
        self.declare_parameter('ref_timeout_sec', 0.5)
        self.declare_parameter('encoder_timeout_sec', 0.25)
        self.declare_parameter('wheel_speed_filter_alpha', 0.5)
        self.declare_parameter('motor_filter_alpha', 0.6)

        self.dt = float(self.get_parameter('control_period_sec').value)
        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.counts_per_rev = int(self.get_parameter('encoder_counts_per_rev').value)
        self.kp_left = float(self.get_parameter('kp_left').value)
        self.ki_left = float(self.get_parameter('ki_left').value)
        self.kd_left = float(self.get_parameter('kd_left').value)
        self.kp_right = float(self.get_parameter('kp_right').value)
        self.ki_right = float(self.get_parameter('ki_right').value)
        self.kd_right = float(self.get_parameter('kd_right').value)
        self.ref_timeout_sec = float(self.get_parameter('ref_timeout_sec').value)
        self.encoder_timeout_sec = float(self.get_parameter('encoder_timeout_sec').value)
        self.wheel_speed_filter_alpha = float(self.get_parameter('wheel_speed_filter_alpha').value)
        self.motor_filter_alpha = float(self.get_parameter('motor_filter_alpha').value)

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

        # The reference comes from either Zenoh cmd_vel bridge or direct action chunk
        # tracking. Encoder counts come from the Roboclaw node.
        self.create_subscription(
            LeftRightFloat32,
            self.get_parameter('wheel_reference_topic').value,
            self.reference_callback,
            10,
        )
        self.create_subscription(
            LeftRightInt32,
            self.get_parameter('encoder_counts_topic').value,
            self.encoder_callback,
            10,
        )
        self.publisher = self.create_publisher(
            LeftRightFloat32,
            self.get_parameter('duty_cycle_topic').value,
            10,
        )
        self.timer = self.create_timer(self.dt, self.control_loop)

    def reference_callback(self, msg):
        """Update desired wheel speeds and refresh the command watchdog."""
        self.v_left_ref = float(msg.left)
        self.v_right_ref = float(msg.right)
        self.last_ref_time = self.get_clock().now()

    def encoder_callback(self, msg):
        """Convert encoder delta counts into filtered wheel speed measurements."""
        now = self.get_clock().now()
        encoder_dt = self.dt
        if self.last_encoder_time is not None:
            encoder_dt = max((now - self.last_encoder_time).nanoseconds / 1e9, 1e-6)
        self.last_encoder_time = now

        distance_per_count = (2.0 * np.pi * self.wheel_radius) / self.counts_per_rev
        raw_left = (float(msg.left) * distance_per_count) / encoder_dt
        raw_right = (float(msg.right) * distance_per_count) / encoder_dt
        alpha = self.wheel_speed_filter_alpha
        self.v_left_meas = alpha * raw_left + (1.0 - alpha) * self.v_left_meas
        self.v_right_meas = alpha * raw_right + (1.0 - alpha) * self.v_right_meas

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
        if age > self.ref_timeout_sec:
            # If high-level commands stop, clear references and integral memory so
            # the robot stops instead of continuing on stale accumulated error.
            self.v_left_ref = 0.0
            self.v_right_ref = 0.0
            self.integral_left = 0.0
            self.integral_right = 0.0

        encoder_stale = (
            self.last_encoder_time is None or
            (self.get_clock().now() - self.last_encoder_time).nanoseconds / 1e9 > self.encoder_timeout_sec
        )
        if encoder_stale:
            self.v_left_ref = 0.0
            self.v_right_ref = 0.0
            self.integral_left = 0.0
            self.integral_right = 0.0

        left_error = self.v_left_ref - self.v_left_meas
        right_error = self.v_right_ref - self.v_right_meas
        d_left = (left_error - self.prev_left_error) / self.dt
        d_right = (right_error - self.prev_right_error) / self.dt

        self.integral_left = float(np.clip(self.integral_left + left_error * self.dt, -50.0, 50.0))
        self.integral_right = float(np.clip(self.integral_right + right_error * self.dt, -50.0, 50.0))

        duty_left = (
            0.97 * self.feedforward_left(self.v_left_ref)
            + self.kp_left * left_error
            + self.ki_left * self.integral_left
            + self.kd_left * d_left
        )
        duty_right = (
            0.99 * self.feedforward_right(self.v_right_ref)
            + self.kp_right * right_error
            + self.ki_right * self.integral_right
            + self.kd_right * d_right
        )

        self.prev_left_error = left_error
        self.prev_right_error = right_error

        # Bound to Roboclaw percent duty convention and smooth motor commands to
        # reduce sharp current spikes from sudden model output changes.
        duty_left = float(np.clip(duty_left, -100.0, 100.0))
        duty_right = float(np.clip(duty_right, -100.0, 100.0))
        alpha = self.motor_filter_alpha
        duty_left = alpha * duty_left + (1.0 - alpha) * self.prev_duty_left
        duty_right = alpha * duty_right + (1.0 - alpha) * self.prev_duty_right
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


if __name__ == '__main__':
    main()
