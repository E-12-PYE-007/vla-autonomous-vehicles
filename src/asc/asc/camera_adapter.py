"""Republish /cam (ImageWithSeqNum) as a plain CompressedImage for RViz/rqt."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

from custom_msgs.msg import ImageWithSeqNum

INPUT_TOPIC = '/cam'
OUTPUT_TOPIC = '/cam/view'

_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class CameraAdapter(Node):
    def __init__(self):
        super().__init__('camera_adapter')
        self.pub = self.create_publisher(CompressedImage, OUTPUT_TOPIC, _QOS)
        self.create_subscription(ImageWithSeqNum, INPUT_TOPIC, self._callback, _QOS)
        self.get_logger().info(f'Forwarding {INPUT_TOPIC} → {OUTPUT_TOPIC}')

    def _callback(self, msg: ImageWithSeqNum):
        self.pub.publish(msg.img)


def main(args=None):
    rclpy.init(args=args)
    node = CameraAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
