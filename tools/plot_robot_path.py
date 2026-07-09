#!/usr/bin/env python3
"""Real-time plot of robot pose and VLA action-chunk target waypoints.

Subscribes to:
  odom_pose2d              geometry_msgs/Pose2D          — robot position & heading
  /asyncvla/action_chunk   asclinic_vla_interfaces/ActionChunk — target path

Works with both the sim stack (odom_to_pose2d → odom_pose2d) and the hardware
split stack (encoder_odometry → odom_pose2d).

Usage:
  python3 tools/plot_robot_path.py
  python3 tools/plot_robot_path.py --window-radius 5.0
  python3 tools/plot_robot_path.py --pose-topic /robot/odom_pose2d
"""

import argparse
import math
import threading
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D
from asclinic_vla_interfaces.msg import ActionChunk


# ── tuneable defaults ────────────────────────────────────────────────────────
DEFAULT_TRAIL_LEN   = 500    # past robot positions to keep
DEFAULT_WINDOW_R    = 3.0    # metres of map visible around the robot
UPDATE_INTERVAL_MS  = 100    # matplotlib refresh period
LOOKAHEAD_DIST      = 0.35   # must match action_chunk_pure_pursuit_node param
MIN_FORWARD_X       = 0.02   # must match action_chunk_pure_pursuit_node param
HEADING_ARROW_LEN   = 0.30   # metres


# ── helpers ──────────────────────────────────────────────────────────────────

def _select_lookahead(rel_pts):
    """Replicate ActionChunkPurePursuitNode.select_lookahead().

    rel_pts is a list of (x, y) tuples in robot frame (metres).
    Returns the (x, y) tuple that the pure-pursuit controller targets.
    """
    selected = rel_pts[-1]
    for pt in rel_pts:
        if pt[0] < MIN_FORWARD_X:
            continue
        if math.hypot(pt[0], pt[1]) >= LOOKAHEAD_DIST:
            selected = pt
            break
    return selected


def _robot_to_world(anchor, rel_pts):
    """Transform robot-relative (x, y) pairs into world-frame coordinates.

    Uses the same rotation as OdomActionChunkTrackerNode.action_callback().
    anchor is (x, y, theta) from odom_pose2d at chunk-arrival time.
    """
    ax, ay, atheta = anchor
    cos_a = math.cos(atheta)
    sin_a = math.sin(atheta)
    return [
        (ax + cos_a * rx - sin_a * ry,
         ay + sin_a * rx + cos_a * ry)
        for rx, ry in rel_pts
    ]


# ── ROS node (runs in background thread) ─────────────────────────────────────

class PlotterNode(Node):
    def __init__(self, pose_topic, chunk_topic, trail_len):
        super().__init__('vla_path_plotter')
        self.lock = threading.Lock()

        # Shared state written by ROS callbacks, read by matplotlib.
        self._robot_pose   = None          # (x, y, theta) — latest odom
        self._trail        = deque(maxlen=trail_len)  # [(x, y), ...]
        self._anchor       = None          # robot pose at last chunk arrival
        self._world_chunk  = []            # [(x, y)] world-frame waypoints
        self._lookahead_pt = None          # (x, y) world-frame lookahead

        self.create_subscription(Pose2D,      pose_topic,  self._pose_cb,  10)
        self.create_subscription(ActionChunk, chunk_topic, self._chunk_cb, 10)
        self.get_logger().info(
            f'Plotting  pose={pose_topic}  chunk={chunk_topic}'
        )

    # ── callbacks (called from rclpy executor thread) ────────────────────────

    def _pose_cb(self, msg):
        with self.lock:
            self._robot_pose = (msg.x, msg.y, msg.theta)
            self._trail.append((msg.x, msg.y))
            self._anchor = (msg.x, msg.y, msg.theta)

    def _chunk_cb(self, msg):
        if not msg.relative_poses:
            return
        with self.lock:
            anchor = self._anchor
        if anchor is None:
            return

        rel_pts = [(float(p.x), float(p.y)) for p in msg.relative_poses]
        world_pts = _robot_to_world(anchor, rel_pts)
        lh_rel = _select_lookahead(rel_pts)
        lh_world = _robot_to_world(anchor, [lh_rel])[0]

        with self.lock:
            self._world_chunk  = world_pts
            self._lookahead_pt = lh_world

    # ── snapshot for the plot thread (avoids holding the lock during drawing) ─

    def snapshot(self):
        with self.lock:
            return (
                self._robot_pose,
                list(self._trail),
                list(self._world_chunk),
                self._lookahead_pt,
            )


# ── matplotlib setup & animation ─────────────────────────────────────────────

def build_plot():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('VLA Robot Path (real-time)')
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('y  [m]')

    trail_line,    = ax.plot([], [], 'b-',  lw=1.2, alpha=0.5, label='Robot trail')
    robot_dot,     = ax.plot([], [], 'bo',  ms=8,               label='Robot')
    heading_line,  = ax.plot([], [], 'b-',  lw=2.5)
    chunk_line,    = ax.plot([], [], 'g--', lw=1.5, alpha=0.8,  label='VLA waypoints')
    chunk_dots,    = ax.plot([], [], 'g.',  ms=6)
    lookahead_dot, = ax.plot([], [], 'r*',  ms=14,              label='Lookahead target')

    ax.legend(loc='upper left', fontsize=9)
    return fig, ax, (trail_line, robot_dot, heading_line,
                     chunk_line, chunk_dots, lookahead_dot)


def make_updater(node, ax, artists, window_radius):
    trail_line, robot_dot, heading_line, chunk_line, chunk_dots, lookahead_dot = artists

    def update(_frame):
        pose, trail, chunk, lookahead = node.snapshot()

        if pose is None:
            return artists

        rx, ry, rtheta = pose

        # Scroll the view window to follow the robot.
        r = window_radius
        ax.set_xlim(rx - r, rx + r)
        ax.set_ylim(ry - r, ry + r)

        # Past trajectory trail.
        if trail:
            tx, ty = zip(*trail)
            trail_line.set_data(tx, ty)
        else:
            trail_line.set_data([], [])

        # Robot marker.
        robot_dot.set_data([rx], [ry])

        # Heading indicator (line from robot in direction of theta).
        ex = rx + HEADING_ARROW_LEN * math.cos(rtheta)
        ey = ry + HEADING_ARROW_LEN * math.sin(rtheta)
        heading_line.set_data([rx, ex], [ry, ey])

        # Action-chunk waypoints in world frame.
        if chunk:
            cx, cy = zip(*chunk)
            chunk_line.set_data(cx, cy)
            chunk_dots.set_data(cx, cy)
        else:
            chunk_line.set_data([], [])
            chunk_dots.set_data([], [])

        # Lookahead point the pure-pursuit controller is steering toward.
        if lookahead:
            lookahead_dot.set_data([lookahead[0]], [lookahead[1]])
        else:
            lookahead_dot.set_data([], [])

        return artists

    return update


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Real-time VLA path plotter')
    parser.add_argument('--pose-topic',    default='odom_pose2d',
                        help='geometry_msgs/Pose2D topic for robot odometry')
    parser.add_argument('--chunk-topic',   default='/asyncvla/action_chunk',
                        help='ActionChunk topic from the VLA action head')
    parser.add_argument('--trail-length',  type=int,   default=DEFAULT_TRAIL_LEN,
                        help='Number of past robot positions to display')
    parser.add_argument('--window-radius', type=float, default=DEFAULT_WINDOW_R,
                        help='Metres of map visible around the robot')
    args = parser.parse_args()

    rclpy.init()
    node = PlotterNode(args.pose_topic, args.chunk_topic, args.trail_length)

    ros_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    ros_thread.start()

    fig, ax, artists = build_plot()
    updater = make_updater(node, ax, artists, args.window_radius)
    _anim = FuncAnimation(fig, updater, interval=UPDATE_INTERVAL_MS, blit=False)

    try:
        plt.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
