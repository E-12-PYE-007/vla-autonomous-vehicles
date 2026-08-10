#!/usr/bin/env python3
"""Log the latest sys1 and sys2 action chunks to CSV once per second.

Subscribes to:
- /asyncvla/action_chunk       (sys1: edge-adapter refined trajectory)
- /asyncvla/omni_action_chunk  (sys2: base VLA's own prediction)

Every second the timer snapshots whatever is currently the latest chunk from
each stream and appends long-format rows. Rows are self-describing so the same
CSV can be reloaded with pandas and pivoted for plotting.

CSV columns:
    stamp_sec, source, seq_num, waypoint_idx, x, y, theta
"""

import csv
import os
from datetime import datetime
from threading import Lock

import rclpy
from custom_msgs.msg import ActionChunk
from rclpy.node import Node


SNAPSHOT_RATE_HZ = 1.0


class StoreActionChunksNode(Node):
    def __init__(self):
        super().__init__("store_action_chunks")

        # Relative to the working dir where the node is launched (the ROS ws root).
        self.declare_parameter("output_dir", "data")
        output_dir = self.get_parameter("output_dir").value
        os.makedirs(output_dir, exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"action_chunks_{run_id}.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["stamp_sec", "source", "seq_num", "waypoint_idx", "x", "y", "theta"])
        self._file.flush()

        self._lock = Lock()
        self._latest_sys1 = None
        self._latest_sys2 = None

        self.create_subscription(ActionChunk, "/asyncvla/action_chunk", self._sys1_callback, 10)
        self.create_subscription(ActionChunk, "/asyncvla/omni_action_chunk", self._sys2_callback, 10)
        self.create_timer(1.0 / SNAPSHOT_RATE_HZ, self._snapshot)

        self.get_logger().info(f"Logging action chunks to {self.csv_path}")

    def _sys1_callback(self, msg: ActionChunk):
        with self._lock:
            self._latest_sys1 = msg

    def _sys2_callback(self, msg: ActionChunk):
        with self._lock:
            self._latest_sys2 = msg

    def _snapshot(self):
        stamp_sec = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            sys1_msg = self._latest_sys1
            sys2_msg = self._latest_sys2

        wrote_any = False
        if sys1_msg is not None:
            self._write_chunk(stamp_sec, "sys1", sys1_msg)
            wrote_any = True
        if sys2_msg is not None:
            self._write_chunk(stamp_sec, "sys2", sys2_msg)
            wrote_any = True

        if wrote_any:
            self._file.flush()

    def _write_chunk(self, stamp_sec: float, source: str, msg: ActionChunk):
        for idx, pose in enumerate(msg.relative_poses):
            self._writer.writerow([
                f"{stamp_sec:.6f}",
                source,
                msg.seq_num,
                idx,
                f"{pose.x:.6f}",
                f"{pose.y:.6f}",
                f"{pose.theta:.6f}",
            ])

    def destroy_node(self):
        try:
            self._file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StoreActionChunksNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
