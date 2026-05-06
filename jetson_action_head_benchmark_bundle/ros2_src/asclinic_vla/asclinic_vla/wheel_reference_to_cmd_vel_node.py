#!/usr/bin/env python3
"""Convert ASClinic wheel velocity references into simulator cmd_vel commands."""

import rclpy
from asclinic_vla_interfaces.msg import LeftRightFloat32
from geometry_msgs.msg import Twist
from rclpy.node import Node


class WheelReferenceToCmdVelNode(Node):
    """Bridge the ASClinic tracker output to the Gazebo DiffDrive input."""

    def __init__(self):
        super().__init__('wheel_reference_to_cmd_vel')

        self.declare_parameter('wheel_reference_topic', 'wheel_velocity_reference')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('wheel_base', 0.22)

        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.publisher = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            10,
        )
        self.create_subscription(
            LeftRightFloat32,
            self.get_parameter('wheel_reference_topic').value,
            self.wheel_reference_callback,
            10,
        )

        self.get_logger().info(
            f'Converting {self.get_parameter("wheel_reference_topic").value} to '
            f'{self.get_parameter("cmd_vel_topic").value}'
        )

    def wheel_reference_callback(self, msg):
        """Convert differential wheel speeds into linear/angular velocity."""
        left = float(msg.left)
        right = float(msg.right)

        twist = Twist()
        twist.linear.x = 0.5 * (left + right)
        twist.angular.z = (right - left) / self.wheel_base
        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WheelReferenceToCmdVelNode()
    try:
        rclpy.spin(node)
    finally:
        node.publisher.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
