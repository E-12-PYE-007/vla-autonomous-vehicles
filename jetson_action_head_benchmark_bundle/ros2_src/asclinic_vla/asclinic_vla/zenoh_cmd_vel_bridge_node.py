#!/usr/bin/env python3
"""Bridge Jetson action-head velocity commands back into the ROS wheel pipeline.

The action head publishes `vla/cmd_vel` as either:
- the current script's binary struct.pack("ddd", timestamp, linear_x, angular_z)
- or a JSON object with linear_x/angular_z, which is useful for future tools.

This node converts body velocity into differential-drive wheel velocity
references for the ASClinic wheel PID controller.
"""

import json
import struct
import threading

import rclpy
from asclinic_vla_interfaces.msg import LeftRightFloat32
from rclpy.node import Node

from asclinic_vla.zenoh_utils import open_zenoh_session


class ZenohCmdVelBridgeNode(Node):
    def __init__(self):
        super().__init__('zenoh_cmd_vel_bridge')

        # wheel_base controls the differential-drive conversion:
        # left = v - L*w/2, right = v + L*w/2.
        self.declare_parameter('wheel_reference_topic', 'wheel_velocity_reference')
        self.declare_parameter('zenoh_cmd_vel_key', 'vla/cmd_vel')
        self.declare_parameter('zenoh_connect_endpoints', ['tcp/127.0.0.1:7447'])
        self.declare_parameter('zenoh_listen_endpoints', [])
        self.declare_parameter('wheel_base', 0.22)
        self.declare_parameter('command_timeout_sec', 0.75)
        self.declare_parameter('timeout_check_rate_hz', 10.0)

        self.zenoh_key = self.get_parameter('zenoh_cmd_vel_key').value
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.command_timeout_sec = float(self.get_parameter('command_timeout_sec').value)
        self.last_command_time = None
        self.seq_num = 1
        self.lock = threading.Lock()

        # Publish wheel velocity references, not duty cycles. The wheel PID remains
        # responsible for using encoder feedback to command the Roboclaw safely.
        self.ros_publisher = self.create_publisher(
            LeftRightFloat32,
            self.get_parameter('wheel_reference_topic').value,
            10,
        )

        self.zenoh_session = open_zenoh_session(
            self.get_parameter('zenoh_connect_endpoints').value,
            self.get_parameter('zenoh_listen_endpoints').value,
        )
        self.subscriber = self.zenoh_session.declare_subscriber(
            self.zenoh_key,
            self.cmd_vel_callback,
        )

        check_rate = float(self.get_parameter('timeout_check_rate_hz').value)
        self.timer = self.create_timer(1.0 / max(check_rate, 1.0), self.timeout_check)
        self.get_logger().info(f'Bridging Zenoh {self.zenoh_key} to wheel references')

    @staticmethod
    def payload_bytes(msg):
        """Return raw Zenoh payload bytes across Zenoh Python API versions."""
        return msg.payload.to_bytes()

    def parse_cmd_vel(self, payload):
        """Decode the action-head command payload into timestamp, v, and omega."""
        if len(payload) == struct.calcsize('ddd'):
            timestamp, linear_x, angular_z = struct.unpack('ddd', payload)
            return timestamp, linear_x, angular_z

        data = json.loads(payload.decode('utf-8'))
        return (
            float(data.get('timestamp', data.get('t_cmd', 0.0))),
            float(data['linear_x']),
            float(data['angular_z']),
        )

    def cmd_vel_callback(self, msg):
        """Convert each received body command into left/right wheel velocity refs."""
        try:
            _, linear_x, angular_z = self.parse_cmd_vel(self.payload_bytes(msg))
        except Exception as exc:
            self.get_logger().warn(f'Failed to parse Zenoh cmd_vel payload: {exc}')
            return

        left = linear_x - 0.5 * self.wheel_base * angular_z
        right = linear_x + 0.5 * self.wheel_base * angular_z
        with self.lock:
            self.last_command_time = self.get_clock().now()
        self.publish_wheel_reference(left, right)

    def timeout_check(self):
        """Stop the robot if the action head stops publishing fresh commands."""
        with self.lock:
            last_command_time = self.last_command_time

        if last_command_time is None:
            return

        age = (self.get_clock().now() - last_command_time).nanoseconds / 1e9
        if age > self.command_timeout_sec:
            self.publish_wheel_reference(0.0, 0.0)

    def publish_wheel_reference(self, left, right):
        """Publish one left/right wheel velocity reference message."""
        msg = LeftRightFloat32()
        msg.left = float(left)
        msg.right = float(right)
        msg.seq_num = self.seq_num
        self.ros_publisher.publish(msg)
        self.seq_num += 1

    def destroy_node(self):
        """Command zero wheel speed before shutting down the bridge."""
        self.publish_wheel_reference(0.0, 0.0)
        if hasattr(self, 'zenoh_session'):
            self.zenoh_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZenohCmdVelBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
