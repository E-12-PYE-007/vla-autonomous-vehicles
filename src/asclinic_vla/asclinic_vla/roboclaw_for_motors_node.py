#!/usr/bin/env python3
"""Roboclaw motor driver node for ASClinic differential-drive hardware.

This node is intentionally the only node that touches the motor controller.
Everything upstream publishes normalized duty-cycle or velocity messages.
"""

import rclpy
from asclinic_vla_interfaces.msg import LeftRightFloat32, LeftRightInt32
from rclpy.node import Node

PERCENT_TO_ROBOCLAW = 327.67


class RoboclawForMotorsNode(Node):
    def __init__(self):
        super().__init__('roboclaw_for_motors')

        # dry_run lets us test the whole ROS/VLA/Zenoh stack without opening the
        # serial port or sending commands to physical motors.
        self.declare_parameter('duty_cycle_topic', 'set_motor_duty_cycle')
        self.declare_parameter('current_duty_cycle_topic', 'current_motor_duty_cycle')
        self.declare_parameter('encoder_counts_topic', 'encoder_counts')
        self.declare_parameter('roboclaw_usb_port', '/dev/ttyACM0')
        self.declare_parameter('roboclaw_baudrate', 38400)
        self.declare_parameter('roboclaw_address', 128)
        self.declare_parameter('encoder_period_sec', 0.1)
        self.declare_parameter('max_duty_cycle', 100.0)
        # Match the original ASClinic hardware defaults:
        # M1 is the left motor and is mounted/wired in reverse, so commands and
        # encoder deltas are multiplied by -1. M2/right stays positive.
        self.declare_parameter('left_motor_multiplier', -1.0)
        self.declare_parameter('right_motor_multiplier', 1.0)
        self.declare_parameter('dry_run', False)

        self.address = int(self.get_parameter('roboclaw_address').value)
        self.max_duty_cycle = float(self.get_parameter('max_duty_cycle').value)
        self.left_multiplier = float(self.get_parameter('left_motor_multiplier').value)
        self.right_multiplier = float(self.get_parameter('right_motor_multiplier').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.prev_left_encoder = None
        self.prev_right_encoder = None
        self.seq_num = 1
        self.encoder_seq_num = 1
        self.roboclaw = None
        self.connected = False

        if not self.dry_run:
            self.connect_roboclaw()
        else:
            self.get_logger().warn('Roboclaw node running in dry_run mode')

        self.create_subscription(
            LeftRightFloat32,
            self.get_parameter('duty_cycle_topic').value,
            self.drive_motors_callback,
            1,
        )
        self.current_duty_publisher = self.create_publisher(
            LeftRightFloat32,
            self.get_parameter('current_duty_cycle_topic').value,
            10,
        )
        self.encoder_publisher = self.create_publisher(
            LeftRightInt32,
            self.get_parameter('encoder_counts_topic').value,
            10,
        )
        self.encoder_timer = self.create_timer(
            float(self.get_parameter('encoder_period_sec').value),
            self.publish_encoder_delta,
        )

    def connect_roboclaw(self):
        """Open the Basicmicro/Roboclaw serial connection."""
        try:
            from basicmicro import Basicmicro
        except ImportError as exc:
            raise RuntimeError('Install basicmicro on the robot to use Roboclaw hardware') from exc

        port = self.get_parameter('roboclaw_usb_port').value
        baudrate = int(self.get_parameter('roboclaw_baudrate').value)
        self.roboclaw = Basicmicro(port, baudrate)
        self.connected = bool(self.roboclaw.Open())
        if not self.connected:
            self.get_logger().warn(
                f'Failed to open Roboclaw on {port} at {baudrate}; motor commands will be ignored'
            )
            return
        self.get_logger().info(f'Connected to Roboclaw on {port}')

    @staticmethod
    def clamp(value, low, high):
        """Clamp motor duty cycle to the configured safety limit."""
        return max(low, min(high, value))

    def drive_motors_callback(self, msg):
        """Apply sign multipliers, command Roboclaw, and report actual duty sent."""
        duty_left = self.clamp(float(msg.left) * self.left_multiplier, -self.max_duty_cycle, self.max_duty_cycle)
        duty_right = self.clamp(float(msg.right) * self.right_multiplier, -self.max_duty_cycle, self.max_duty_cycle)

        if self.connected:
            rc_left = int(duty_left * PERCENT_TO_ROBOCLAW)
            rc_right = int(duty_right * PERCENT_TO_ROBOCLAW)
            if not self.roboclaw.DutyM1M2(self.address, rc_left, rc_right):
                self.get_logger().warn('Roboclaw did not acknowledge DutyM1M2 command')

        out = LeftRightFloat32()
        out.left = float(duty_left)
        out.right = float(duty_right)
        out.seq_num = self.seq_num
        self.current_duty_publisher.publish(out)
        self.seq_num += 1

    def read_encoder_pair(self):
        """Read signed absolute encoder counts from Roboclaw M1 and M2.

        The original ASClinic driver uses GetEncoders so both channels are read
        from the same Roboclaw transaction. It also converts unsigned 32-bit
        values into signed integers before computing deltas.
        """
        try:
            result = self.roboclaw.GetEncoders(self.address)
        except Exception as exc:
            self.get_logger().warn(f'Roboclaw encoder read failed: {exc}')
            return None

        if not result[0]:
            return None

        left = result[1] if result[1] < 0x80000000 else result[1] - 0x100000000
        right = result[2] if result[2] < 0x80000000 else result[2] - 0x100000000
        return int(left), int(right)

    def publish_encoder_delta(self):
        """Publish encoder count deltas for the wheel PID measurement update."""
        if not self.connected:
            return

        encoders = self.read_encoder_pair()
        if encoders is None:
            self.get_logger().warn('Failed to read Roboclaw encoders')
            return

        left, right = encoders
        if self.prev_left_encoder is None:
            # First read establishes a baseline; subsequent reads publish deltas.
            self.prev_left_encoder = left
            self.prev_right_encoder = right
            return

        msg = LeftRightInt32()
        msg.left = int((left - self.prev_left_encoder) * self.left_multiplier)
        msg.right = int((right - self.prev_right_encoder) * self.right_multiplier)
        msg.seq_num = self.encoder_seq_num
        self.encoder_publisher.publish(msg)
        self.encoder_seq_num += 1

        self.prev_left_encoder = left
        self.prev_right_encoder = right

    def destroy_node(self):
        """Stop both motors before shutting down the node."""
        if self.connected:
            try:
                self.roboclaw.DutyM1M2(self.address, 0, 0)
                if hasattr(self.roboclaw, 'close'):
                    self.roboclaw.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoboclawForMotorsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
