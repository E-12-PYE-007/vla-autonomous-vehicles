#!/usr/bin/env python3
"""Bridge Jetson action-head pose chunks from Zenoh into ROS ActionChunk."""

import json

import rclpy
from asclinic_vla_interfaces.msg import ActionChunk
from geometry_msgs.msg import Pose2D
from rclpy.node import Node

from asclinic_vla.zenoh_utils import open_zenoh_session


class ZenohActionChunkBridgeNode(Node):
    def __init__(self):
        super().__init__('zenoh_action_chunk_bridge')

        self.declare_parameter('action_topic', '/asyncvla/action_chunk')
        self.declare_parameter('zenoh_action_chunk_key', 'vla/action_chunk')
        self.declare_parameter('zenoh_connect_endpoints', 'tcp/127.0.0.1:7447')
        self.declare_parameter('zenoh_listen_endpoints', '')

        self.zenoh_key = self.get_parameter('zenoh_action_chunk_key').value
        self.fallback_seq_num = 1
        self.publisher = self.create_publisher(
            ActionChunk,
            self.get_parameter('action_topic').value,
            10,
        )

        connect_endpoints = self.get_parameter('zenoh_connect_endpoints').value
        listen_endpoints = self.get_parameter('zenoh_listen_endpoints').value

        if isinstance(connect_endpoints, str):
            connect_endpoints = [connect_endpoints] if connect_endpoints else []

        if isinstance(listen_endpoints, str):
            listen_endpoints = [listen_endpoints] if listen_endpoints else []

        self.zenoh_session = open_zenoh_session(
            connect_endpoints,
            listen_endpoints,
        )
        
        self.subscriber = self.zenoh_session.declare_subscriber(
            self.zenoh_key,
            self.action_chunk_callback,
        )
        self.get_logger().info(f'Bridging Zenoh {self.zenoh_key} to /asyncvla/action_chunk')

    def action_chunk_callback(self, msg):
        """Decode JSON relative poses and publish a ROS ActionChunk."""
        try:
            payload = json.loads(msg.payload.to_bytes().decode('utf-8'))
            relative_poses = payload['relative_poses']
        except Exception as exc:
            self.get_logger().warn(f'Failed to decode action chunk payload: {exc}')
            return

        out = ActionChunk()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = payload.get('frame', 'robot')
        out.seq_num = int(payload.get('seq_num', self.fallback_seq_num))

        for item in relative_poses:
            pose = Pose2D()
            pose.x = float(item[0])
            pose.y = float(item[1])
            pose.theta = float(item[2])
            out.relative_poses.append(pose)

        self.publisher.publish(out)
        self.fallback_seq_num = out.seq_num + 1

    def destroy_node(self):
        if hasattr(self, 'zenoh_session'):
            self.zenoh_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ZenohActionChunkBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
