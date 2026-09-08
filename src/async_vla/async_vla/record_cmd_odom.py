#!/usr/bin/env python3
"""Record /cmd_vel (commanded) and /sim_odom (measured) to a single CSV, aligned
by timestep.

Both topics arrive at their own rates and /cmd_vel is a bare geometry_msgs/Twist
with no header, so rather than trust either message stamp, a timer snapshots the
latest sample from each stream at a fixed rate and writes one row per tick. The
row's stamp_sec is the node clock (same epoch for every column), which keeps the
commanded and measured columns on a common timeline you can diff or plot directly.

Launched automatically by sim_async.launch.py when save_actions:=true, alongside
store_action_chunks. Can also be run on its own:
        ros2 run async_vla record_cmd_odom
        ros2 run async_vla record_cmd_odom --ros-args -p rate_hz:=50.0 -p odom_topic:=/odom

Columns:
    stamp_sec        node-clock time of the snapshot
    cmd_linear_x     /cmd_vel linear.x   (m/s, commanded)
    cmd_angular_z    /cmd_vel angular.z  (rad/s, commanded)
    odom_x, odom_y   /sim_odom pose position (m)
    odom_theta       yaw from /sim_odom orientation (rad, wrapped to [-pi, pi])
    odom_linear_x    /sim_odom twist.linear.x   (m/s, measured)
    odom_angular_z   /sim_odom twist.angular.z  (rad/s, measured)

Empty cells mean no message had arrived on that topic yet at snapshot time.
"""

import csv
import math
import os
from datetime import datetime
from threading import Lock

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class RecordCmdOdomNode(Node):
    def __init__(self):
        super().__init__("record_cmd_odom")

        # Relative to the working dir where the node is launched (the ROS ws root).
        self.declare_parameter("output_dir", "data")
        output_dir = self.get_parameter("output_dir").value
        os.makedirs(output_dir, exist_ok=True)

        self.declare_parameter("rate_hz", 20.0)
        rate_hz = float(self.get_parameter("rate_hz").value)

        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/sim_odom")
        cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"cmd_odom_{run_id}.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "stamp_sec",
            "cmd_linear_x", "cmd_angular_z",
            "odom_x", "odom_y", "odom_theta",
            "odom_linear_x", "odom_angular_z",
        ])
        self._file.flush()

        self._lock = Lock()
        self._latest_cmd = None
        self._latest_odom = None

        self.create_subscription(Twist, cmd_topic, self._cmd_callback, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 10)
        self.create_timer(1.0 / rate_hz, self._snapshot)

        self.get_logger().info(
            f"Recording {cmd_topic} + {odom_topic} at {rate_hz:.0f} Hz to {self.csv_path}"
        )

    def _cmd_callback(self, msg: Twist):
        with self._lock:
            self._latest_cmd = msg

    def _odom_callback(self, msg: Odometry):
        with self._lock:
            self._latest_odom = msg

    def _snapshot(self):
        stamp_sec = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            cmd = self._latest_cmd
            odom = self._latest_odom

        row = [f"{stamp_sec:.6f}"]

        if cmd is not None:
            row += [f"{cmd.linear.x:.6f}", f"{cmd.angular.z:.6f}"]
        else:
            row += ["", ""]

        if odom is not None:
            p = odom.pose.pose.position
            q = odom.pose.pose.orientation
            theta = wrap_to_pi(2.0 * math.atan2(q.z, q.w))
            t = odom.twist.twist
            row += [
                f"{p.x:.6f}", f"{p.y:.6f}", f"{theta:.6f}",
                f"{t.linear.x:.6f}", f"{t.angular.z:.6f}",
            ]
        else:
            row += ["", "", "", "", ""]

        self._writer.writerow(row)
        self._file.flush()

    def destroy_node(self):
        try:
            self._file.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecordCmdOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
