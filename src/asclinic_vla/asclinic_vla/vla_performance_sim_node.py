#!/usr/bin/env python3
"""Passive VLA performance monitor for Gazebo simulation runs."""

import csv
import json
import math
import os
import re
import struct
import threading
import time
from pathlib import Path

import rclpy
from asclinic_vla_interfaces.msg import ActionChunk, LeftRightFloat32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from asclinic_vla.zenoh_utils import open_zenoh_session


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class VLASimPerformanceMonitor(Node):
    """Collects latency, command, action-chunk, and path metrics during sim."""

    def __init__(self):
        super().__init__('vla_sim_performance_monitor')

        self.declare_parameter('output_dir', 'vla_perf_logs/sim')
        self.declare_parameter('input_csv', '')
        self.declare_parameter('mode', 'record')
        self.declare_parameter('session_name', '')
        self.declare_parameter('summary_period_sec', 5.0)
        self.declare_parameter('use_zenoh', True)
        self.declare_parameter('zenoh_connect_endpoints', ['tcp/127.0.0.1:7447'])
        self.declare_parameter('zenoh_listen_endpoints', [])
        self.declare_parameter('camera_zenoh_key', 'camera/img_compressed')
        self.declare_parameter('vla_actions_key', 'vla/actions')
        self.declare_parameter('cmd_vel_zenoh_key', 'vla/cmd_vel')
        self.declare_parameter('action_chunk_zenoh_key', 'vla/action_chunk')
        self.declare_parameter('camera_topic', '/cam')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('wheel_reference_topic', 'wheel_velocity_reference')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('action_topic', '/asyncvla/action_chunk')

        self.mode = str(self.get_parameter('mode').value)
        self.output_dir = self.make_output_dir()
        if self.mode == 'plot':
            self.create_timer(0.1, self.plot_and_shutdown)
            return

        self.csv_path = self.output_dir / 'events.csv'
        self.summary_path = self.output_dir / 'summary.txt'
        self.csv_file = self.csv_path.open('w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                'local_time', 'event', 'seq', 'source_time', 'latency_sec',
                'stage_latency_sec', 'x', 'y', 'theta', 'linear_x',
                'angular_z', 'left', 'right', 'count', 'path_length',
                'extra',
            ],
        )
        self.csv_writer.writeheader()

        self.lock = threading.Lock()
        self.latest_camera_t = None
        self.latest_vla_t = None
        self.last_cmd_t = None
        self.ros_camera_count = 0
        self.zenoh_camera_count = 0
        self.vla_action_count = 0
        self.cmd_count = 0
        self.action_chunk_count = 0
        self.odom_path = []
        self.cmd_series = []
        self.wheel_ref_series = []
        self.latency_series = []
        self.chunk_series = []

        self.create_subscription(Image, self.get_parameter('camera_topic').value, self.camera_callback, 10)
        self.create_subscription(Twist, self.get_parameter('cmd_vel_topic').value, self.cmd_vel_callback, 10)
        self.create_subscription(
            LeftRightFloat32,
            self.get_parameter('wheel_reference_topic').value,
            self.wheel_ref_callback,
            10,
        )
        self.create_subscription(Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        self.create_subscription(ActionChunk, self.get_parameter('action_topic').value, self.action_chunk_callback, 10)

        self.zenoh_session = None
        if bool(self.get_parameter('use_zenoh').value):
            self.open_zenoh()

        summary_period = float(self.get_parameter('summary_period_sec').value)
        self.create_timer(max(summary_period, 1.0), self.write_summary)

        self.get_logger().info(f'Writing sim VLA performance logs to {self.output_dir}')

    def make_output_dir(self):
        root = Path(str(self.get_parameter('output_dir').value)).expanduser()
        session_name = str(self.get_parameter('session_name').value).strip()
        if not session_name:
            session_name = time.strftime('%Y%m%d_%H%M%S')
        path = root / session_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def open_zenoh(self):
        self.zenoh_session = open_zenoh_session(
            self.get_parameter('zenoh_connect_endpoints').value,
            self.get_parameter('zenoh_listen_endpoints').value,
        )
        self.zenoh_session.declare_subscriber(
            self.get_parameter('camera_zenoh_key').value,
            self.zenoh_camera_callback,
        )
        self.zenoh_session.declare_subscriber(
            self.get_parameter('vla_actions_key').value,
            self.zenoh_vla_actions_callback,
        )
        self.zenoh_session.declare_subscriber(
            self.get_parameter('cmd_vel_zenoh_key').value,
            self.zenoh_cmd_vel_callback,
        )
        self.zenoh_session.declare_subscriber(
            self.get_parameter('action_chunk_zenoh_key').value,
            self.zenoh_action_chunk_callback,
        )

    def write_event(self, event, **kwargs):
        row = {name: '' for name in self.csv_writer.fieldnames}
        row['local_time'] = f'{time.time():.6f}'
        row['event'] = event
        for key, value in kwargs.items():
            if key in row:
                row[key] = value
            else:
                row['extra'] = f'{row["extra"]} {key}={value}'.strip()
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def camera_callback(self, msg):
        with self.lock:
            self.ros_camera_count += 1
        source_time = stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec or msg.header.stamp.nanosec else ''
        self.write_event('ros_camera', count=self.ros_camera_count, source_time=source_time)

    def cmd_vel_callback(self, msg):
        now = time.time()
        with self.lock:
            self.cmd_series.append((now, float(msg.linear.x), float(msg.angular.z)))
        self.write_event('ros_cmd_vel', linear_x=f'{msg.linear.x:.6f}', angular_z=f'{msg.angular.z:.6f}')

    def wheel_ref_callback(self, msg):
        now = time.time()
        with self.lock:
            self.wheel_ref_series.append((now, float(msg.left), float(msg.right)))
        self.write_event('wheel_reference', seq=msg.seq_num, left=f'{msg.left:.6f}', right=f'{msg.right:.6f}')

    def odom_callback(self, msg):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        theta = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        with self.lock:
            self.odom_path.append((time.time(), x, y, theta))
        self.write_event('odom', x=f'{x:.6f}', y=f'{y:.6f}', theta=f'{theta:.6f}')

    def action_chunk_callback(self, msg):
        poses = [(float(p.x), float(p.y), float(p.theta)) for p in msg.relative_poses]
        path_length = sum(dist2d(poses[i - 1], poses[i]) for i in range(1, len(poses))) if len(poses) > 1 else 0.0
        with self.lock:
            self.action_chunk_count += 1
            self.chunk_series.append((time.time(), len(poses), path_length))
        self.write_event(
            'ros_action_chunk',
            seq=msg.seq_num,
            count=len(poses),
            path_length=f'{path_length:.6f}',
        )

    def payload_text_head(self, msg, size=4096):
        return msg.payload.to_bytes()[:size].decode('utf-8', errors='ignore')

    def zenoh_camera_callback(self, msg):
        text = self.payload_text_head(msg)
        t_camera = self.extract_float(text, 't_camera')
        with self.lock:
            self.zenoh_camera_count += 1
            if t_camera is not None:
                self.latest_camera_t = t_camera
        latency = time.time() - t_camera if t_camera is not None else ''
        self.write_event('zenoh_camera', count=self.zenoh_camera_count, source_time=t_camera or '', latency_sec=latency)

    def zenoh_vla_actions_callback(self, msg):
        text = self.payload_text_head(msg)
        t_vla = self.extract_float(text, 't_vla')
        shape = self.extract_shape(text)
        with self.lock:
            self.vla_action_count += 1
            camera_t = self.latest_camera_t
            if t_vla is not None:
                self.latest_vla_t = t_vla
        stage_latency = t_vla - camera_t if t_vla is not None and camera_t is not None else ''
        if stage_latency != '':
            with self.lock:
                self.latency_series.append((time.time(), 'camera_to_vla', float(stage_latency)))
        self.write_event(
            'zenoh_vla_actions',
            count=self.vla_action_count,
            source_time=t_vla or '',
            stage_latency_sec=stage_latency,
            extra=f'shape={shape}',
        )

    def zenoh_cmd_vel_callback(self, msg):
        payload = msg.payload.to_bytes()
        t_cmd = None
        linear_x = None
        angular_z = None
        if len(payload) == struct.calcsize('ddd'):
            t_cmd, linear_x, angular_z = struct.unpack('ddd', payload)
        else:
            try:
                data = json.loads(payload.decode('utf-8'))
                t_cmd = float(data.get('timestamp', data.get('t_cmd', 0.0)))
                linear_x = float(data['linear_x'])
                angular_z = float(data['angular_z'])
            except Exception:
                pass

        now = time.time()
        with self.lock:
            self.cmd_count += 1
            vla_t = self.latest_vla_t
            last_cmd_t = self.last_cmd_t
            self.last_cmd_t = t_cmd
        vla_to_cmd = t_cmd - vla_t if t_cmd is not None and vla_t is not None else ''
        cmd_period = t_cmd - last_cmd_t if t_cmd is not None and last_cmd_t is not None else ''
        if vla_to_cmd != '':
            with self.lock:
                self.latency_series.append((now, 'vla_to_cmd', float(vla_to_cmd)))
        self.write_event(
            'zenoh_cmd_vel',
            count=self.cmd_count,
            source_time=t_cmd or '',
            latency_sec=(now - t_cmd) if t_cmd is not None else '',
            stage_latency_sec=vla_to_cmd,
            linear_x=f'{linear_x:.6f}' if linear_x is not None else '',
            angular_z=f'{angular_z:.6f}' if angular_z is not None else '',
            extra=f'cmd_period={cmd_period}',
        )

    def zenoh_action_chunk_callback(self, msg):
        try:
            data = json.loads(msg.payload.to_bytes().decode('utf-8'))
            poses = data.get('relative_poses', [])
            t_action_head = data.get('t_action_head', None)
        except Exception:
            return
        path_length = sum(
            dist2d(poses[i - 1], poses[i]) for i in range(1, len(poses))
        ) if len(poses) > 1 else 0.0
        self.write_event(
            'zenoh_action_chunk',
            seq=data.get('seq_num', ''),
            source_time=t_action_head or '',
            count=len(poses),
            path_length=f'{path_length:.6f}',
        )

    @staticmethod
    def extract_float(text, key):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*([-+0-9.eE]+)', text)
        return float(match.group(1)) if match else None

    @staticmethod
    def extract_shape(text):
        match = re.search(r'"shape"\s*:\s*\[([^\]]+)\]', text)
        return '[' + match.group(1).strip() + ']' if match else ''

    def write_summary(self):
        with self.lock:
            distance = sum(
                dist2d((self.odom_path[i - 1][1], self.odom_path[i - 1][2]), (self.odom_path[i][1], self.odom_path[i][2]))
                for i in range(1, len(self.odom_path))
            )
            summary = [
                f'local_time: {time.time():.3f}',
                f'ros_camera_frames: {self.ros_camera_count}',
                f'zenoh_camera_frames: {self.zenoh_camera_count}',
                f'vla_actions: {self.vla_action_count}',
                f'zenoh_cmd_vel: {self.cmd_count}',
                f'ros_action_chunks: {self.action_chunk_count}',
                f'odom_samples: {len(self.odom_path)}',
                f'odom_distance_m: {distance:.3f}',
                f'output_dir: {self.output_dir}',
            ]
        self.summary_path.write_text('\n'.join(summary) + '\n', encoding='utf-8')
        self.get_logger().info(' | '.join(summary[1:6]))

    def generate_plots(self):
        self.generate_plots_from_csv(self.csv_path, self.output_dir)

    def plot_and_shutdown(self):
        input_csv = str(self.get_parameter('input_csv').value).strip()
        csv_path = Path(input_csv).expanduser() if input_csv else self.output_dir / 'events.csv'
        output_dir = csv_path.parent
        self.generate_plots_from_csv(csv_path, output_dir)
        self.get_logger().info(f'Generated plots from {csv_path} into {output_dir}')
        rclpy.shutdown()

    def generate_plots_from_csv(self, csv_path, output_dir):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.get_logger().warn(f'Cannot generate plots because matplotlib is unavailable: {exc}')
            return

        if not csv_path.exists():
            self.get_logger().warn(f'CSV file does not exist: {csv_path}')
            return

        odom_path = []
        cmd_series = []
        latency_series = []
        wheel_ref_series = []
        with csv_path.open('r', newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                try:
                    t = float(row['local_time'])
                except (TypeError, ValueError):
                    continue
                if row['event'] == 'odom' and row['x'] and row['y']:
                    odom_path.append((t, float(row['x']), float(row['y']), float(row['theta'] or 0.0)))
                elif row['event'] == 'ros_cmd_vel' and row['linear_x'] and row['angular_z']:
                    cmd_series.append((t, float(row['linear_x']), float(row['angular_z'])))
                elif row['event'] == 'wheel_reference' and row['left'] and row['right']:
                    wheel_ref_series.append((t, float(row['left']), float(row['right'])))
                if row['stage_latency_sec']:
                    latency_series.append((t, row['event'], float(row['stage_latency_sec'])))

        if odom_path:
            xs = [item[1] for item in odom_path]
            ys = [item[2] for item in odom_path]
            plt.figure()
            plt.plot(xs, ys)
            plt.axis('equal')
            plt.title('Robot Path')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.grid(True)
            plt.savefig(self.output_dir / 'path.png', dpi=150)
            plt.close()

        if cmd_series:
            t0 = cmd_series[0][0]
            ts = [item[0] - t0 for item in cmd_series]
            plt.figure()
            plt.plot(ts, [item[1] for item in cmd_series], label='linear.x')
            plt.plot(ts, [item[2] for item in cmd_series], label='angular.z')
            plt.title('ROS /cmd_vel')
            plt.xlabel('time [s]')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.output_dir / 'cmd_vel.png', dpi=150)
            plt.close()

        if wheel_ref_series:
            t0 = wheel_ref_series[0][0]
            ts = [item[0] - t0 for item in wheel_ref_series]
            plt.figure()
            plt.plot(ts, [item[1] for item in wheel_ref_series], label='left')
            plt.plot(ts, [item[2] for item in wheel_ref_series], label='right')
            plt.title('Wheel Velocity References')
            plt.xlabel('time [s]')
            plt.ylabel('m/s')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.output_dir / 'wheel_references.png', dpi=150)
            plt.close()

        if latency_series:
            plt.figure()
            for label in sorted(set(item[1] for item in latency_series)):
                points = [item for item in latency_series if item[1] == label]
                t0 = points[0][0]
                plt.plot([item[0] - t0 for item in points], [item[2] for item in points], '.', label=label)
            plt.title('Zenoh Stage Latencies')
            plt.xlabel('time [s]')
            plt.ylabel('latency [s]')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.output_dir / 'latency.png', dpi=150)
            plt.close()

    def destroy_node(self):
        if self.mode != 'plot':
            self.write_summary()
            self.csv_file.close()
            if self.zenoh_session is not None:
                self.zenoh_session.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VLASimPerformanceMonitor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
