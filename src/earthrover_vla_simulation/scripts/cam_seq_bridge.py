#!/usr/bin/python3
"""Republish Gazebo camera frames as ImageWithSeqNum, the way the real camera does.

Deliberately not `env python3`: this repo is normally driven from shells with a conda
env active, and rclpy's C extension is built against the ROS distro's interpreter, so
`env` picks the wrong one and the node dies on import. It needs nothing beyond what
ROS itself installs, so pointing straight at the system interpreter is safe here.

ros_gz_bridge can only emit standard ROS types, so in sim /cam_raw arrives as a plain
sensor_msgs/Image with no sequence number. On hardware asc/camera_capture.py stamps
every frame with one, and the inference nodes pair a hidden state with the exact frame
it was computed from by looking that number up.

Without this node each consumer has to invent the missing number itself, and two
consumers inventing it independently do not agree: sys1 counts every frame at ~10 Hz
while sys2, whose callback blocks through inference on a depth-1 queue, counts only the
~4 Hz it actually processes. The two numberings drift apart without bound, sys1's
lookup misses forever, and it silently stops publishing action chunks.

Assigning the number once, at the only publisher, makes that divergence impossible to
express rather than merely fixed.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from custom_msgs.msg import ImageWithSeqNum


class CamSeqBridgeNode(Node):
    def __init__(self):
        super().__init__("cam_seq_bridge")

        self.seq_num = 0
        self.publisher = self.create_publisher(ImageWithSeqNum, "/cam", 10)
        # SENSOR_DATA to match the gz bridge's publisher QoS, otherwise the
        # subscription is incompatible and no frames arrive.
        self.create_subscription(Image, "/cam_raw", self.republish, qos_profile_sensor_data)

        self.get_logger().info("Stamping /cam_raw -> /cam with sequence numbers")

    def republish(self, msg: Image):
        """Wrap one raw frame, numbering it in arrival order."""
        wrapped = ImageWithSeqNum()
        wrapped.header = msg.header
        wrapped.img = msg
        wrapped.img_seq_num = self.seq_num
        self.publisher.publish(wrapped)
        self.seq_num += 1


def main(args=None):
    rclpy.init(args=args)
    node = CamSeqBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
