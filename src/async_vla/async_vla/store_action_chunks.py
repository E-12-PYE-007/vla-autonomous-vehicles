#!/usr/bin/env python3
"""Log the latest sys1 and sys2 action chunks to CSV once per second.

Subscribes to:
- /asyncvla/action_chunk       (sys1: edge-adapter refined trajectory)
- /asyncvla/omni_action_chunk  (sys2: base VLA's own prediction)
- /cam                         (the frame each chunk's seq_num was computed from)

Every second the timer snapshots whatever is currently the latest chunk from
each stream and appends long-format rows. Rows are self-describing so the same
CSV can be reloaded with pandas and pivoted for plotting.

The /cam frame matching a chunk's seq_num (chunk.seq_num == img_seq_num, see
cam_seq_bridge.py) is saved once to <output_dir>/images_<run_id>/seq_<N>.jpg,
so the same run's action-chunk plot can show what the model actually saw.

Three frames span one sys2 inference, all saved the same way:
    seq_num          what went into sys2, and sys1's hidden state input
    end_img_seq_num  newest frame once sys2's forward pass returned
    curr_img_seq_num what sys1 actually paired with (fresher still; sys1 is faster)
sys2 sets curr_img_seq_num == seq_num, since it only conditions on one frame.

Writes two files:
    action_chunks_<run_id>.csv
        stamp_sec, source, seq_num, waypoint_idx, x, y, theta, image_path,
        curr_image_path, end_image_path, sys2_inference_ms, goal
    frame_times_<run_id>.csv
        img_seq_num, arrival_sec -- join on img_seq_num to turn a gap between
        sequence numbers into an actual elapsed time.
"""

import csv
import os
from collections import deque
from datetime import datetime
from threading import Lock

import cv2
import rclpy
from cv_bridge import CvBridge
from custom_msgs.msg import ActionChunk, ImageWithSeqNum
from rclpy.node import Node
from sensor_msgs.msg import Image


SNAPSHOT_RATE_HZ = 1.0
IMAGE_BUFFER_SIZE = 100  # frames of /cam to keep around, matches sys1's window


def _seq_num_from_stamp(stamp) -> int:
    """Per-frame id derived from a Header stamp, in milliseconds.

    Isaac's raw sensor_msgs/Image on /vla/cam carries no seq num like ImageWithSeqNum
    does, so it is derived from the stamp the same way sys1/sys2 do it, keeping the
    buffer keys here aligned with the seq_num on the chunks they publish. Must match
    sys1/sys2's copy of this helper.
    """
    return stamp.sec * 1000 + stamp.nanosec // 1_000_000


class StoreActionChunksNode(Node):
    def __init__(self):
        super().__init__("store_action_chunks")

        # Relative to the working dir where the node is launched (the ROS ws root).
        self.declare_parameter("output_dir", "data")
        output_dir = self.get_parameter("output_dir").value
        os.makedirs(output_dir, exist_ok=True)

        # Same language goal passed to sys2, logged alongside the chunks it produced.
        self.declare_parameter("goal", "")
        self.goal_text = self.get_parameter("goal").get_parameter_value().string_value

        # Which camera topic the chunks were computed from: isaac uses raw Image on
        # /vla/cam, everything else ImageWithSeqNum on /cam. Matches sys1/sys2.
        self.declare_parameter("sim", "")
        self.sim = self.get_parameter("sim").get_parameter_value().string_value

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"action_chunks_{run_id}.csv")
        self.images_dir = os.path.join(output_dir, f"images_{run_id}")
        os.makedirs(self.images_dir, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "stamp_sec", "source", "seq_num", "waypoint_idx", "x", "y", "theta",
            "image_path", "curr_image_path", "end_image_path", "sys2_inference_ms", "goal",
        ])
        self._file.flush()

        # seq_num identifies frames but says nothing about elapsed time; join on
        # img_seq_num to turn a gap between sequence numbers into a duration.
        self.frames_csv_path = os.path.join(output_dir, f"frame_times_{run_id}.csv")
        self._frames_file = open(self.frames_csv_path, "w", newline="")
        self._frames_writer = csv.writer(self._frames_file)
        self._frames_writer.writerow(["img_seq_num", "arrival_sec"])
        self._frames_file.flush()

        self._lock = Lock()
        self._latest_sys1 = None
        self._latest_sys2 = None

        self._bridge = CvBridge()
        self._img_lock = Lock()
        self._img_buffer = {}
        self._img_buffer_keys = deque(maxlen=IMAGE_BUFFER_SIZE)
        self._saved_img_seqs = set()

        self.create_subscription(ActionChunk, "/asyncvla/action_chunk", self._sys1_callback, 10)
        self.create_subscription(ActionChunk, "/asyncvla/omni_action_chunk", self._sys2_callback, 10)
        if self.sim == "isaac":
            self.create_subscription(Image, "/vla/cam", self._isaac_img_callback, 10)
            self.get_logger().info("Logging frames from /vla/cam (Image, isaac)")
        else:
            self.create_subscription(ImageWithSeqNum, "/cam", self._img_callback, 10)
            self.get_logger().info("Logging frames from /cam (ImageWithSeqNum)")
        self.create_timer(1.0 / SNAPSHOT_RATE_HZ, self._snapshot)

        self.get_logger().info(f"Logging action chunks to {self.csv_path}")
        self.get_logger().info(f"Logging chunk images to {self.images_dir}")

    def _sys1_callback(self, msg: ActionChunk):
        with self._lock:
            self._latest_sys1 = msg

    def _sys2_callback(self, msg: ActionChunk):
        with self._lock:
            self._latest_sys2 = msg

    def _img_callback(self, msg: ImageWithSeqNum):
        frame = self._bridge.imgmsg_to_cv2(msg.img, desired_encoding="bgr8")
        self._store_frame(frame, msg.img_seq_num)

    def _isaac_img_callback(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._store_frame(frame, _seq_num_from_stamp(msg.header.stamp))

    def _store_frame(self, frame, seq_num: int):
        # Node clock, matching stamp_sec. Not msg.header.stamp, which is sim time here
        # while this node runs with use_sim_time unset.
        arrival_sec = self.get_clock().now().nanoseconds / 1e9
        with self._img_lock:
            if len(self._img_buffer_keys) == IMAGE_BUFFER_SIZE:
                self._img_buffer.pop(self._img_buffer_keys[0], None)
            self._img_buffer_keys.append(seq_num)
            self._img_buffer[seq_num] = frame
        self._frames_writer.writerow([seq_num, f"{arrival_sec:.6f}"])
        self._frames_file.flush()

    def _save_chunk_image(self, seq_num: int) -> str:
        """Save the /cam frame for seq_num once, returning its path (or "" if unavailable)."""
        image_path = os.path.join(self.images_dir, f"seq_{seq_num}.jpg")
        if seq_num in self._saved_img_seqs:
            return image_path

        with self._img_lock:
            frame = self._img_buffer.get(seq_num)
        if frame is None:
            return ""

        cv2.imwrite(image_path, frame)
        self._saved_img_seqs.add(seq_num)
        return image_path

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
        image_path = self._save_chunk_image(msg.seq_num)
        curr_image_path = self._save_chunk_image(msg.curr_img_seq_num)
        end_image_path = self._save_chunk_image(msg.end_img_seq_num)
        for idx, pose in enumerate(msg.relative_poses):
            self._writer.writerow([
                f"{stamp_sec:.6f}",
                source,
                msg.seq_num,
                idx,
                f"{pose.x:.6f}",
                f"{pose.y:.6f}",
                f"{pose.theta:.6f}",
                image_path,
                curr_image_path,
                end_image_path,
                f"{msg.sys2_inference_ms:.3f}",
                self.goal_text,
            ])

    def destroy_node(self):
        try:
            self._file.close()
            self._frames_file.close()
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
