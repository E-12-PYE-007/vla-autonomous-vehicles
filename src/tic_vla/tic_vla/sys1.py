#!/usr/bin/env python3

import math
import os

import torch
import rclpy
from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from custom_msgs.msg import ActionChunk, TicKVCache, TicImageTokens, TicLatencyMetadata

from ticvla.models.ticvla import ActionExpert


ACTION_HORIZON_STEPS  = 30            # 30 steps at 10 Hz = 3 seconds
VLM_HIDDEN_SIZE       = 896           # Qwen2.5-0.5B hidden dim (InternVL3-1B)
ACTION_EXPERT_HIDDEN  = 512
ACTION_EXPERT_LAYERS  = 3
KV_NUM_HEADS          = 2             # InternVL3-1B last-layer KV heads
KV_HEAD_DIM           = 64            # InternVL3-1B KV head dim
KV_CACHE_FEAT_DIM     = KV_NUM_HEADS * KV_HEAD_DIM  # 128
NUM_IMAGE_TOKENS      = 256           # InternViT-300M tokens per frame
CONTROL_RATE_HZ       = 10.0
DEVICE_TYPE           = "cuda"

class Sys1(Node):
    def __init__(self):
        super().__init__("sys1")
        self.get_logger().info("[TIC-VLA Sys1] Initialising...")

        self.declare_parameter("checkpoint_path", "")
        self.checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value

        self.device = torch.device(DEVICE_TYPE)
        self.action_expert = self.load_action_expert()

        self.kv_values = None           # (1, KV_NUM_HEADS, seq_len, KV_HEAD_DIM) bfloat16
        self.kv_job_id = -1
        self.kv_job_stamp = None        # float seconds, wall time when sys2 started this job
        self.kv_job_x = 0.0
        self.kv_job_y = 0.0
        self.kv_job_theta = 0.0
        self.image_tokens = None        # (1, NUM_IMAGE_TOKENS, VLM_HIDDEN_SIZE) bfloat16
        self.current_pose = None        # Pose2D
        self.current_twist = None       # Twist (linear.x=vx, linear.y=vy, angular.z=omega_z)

        self.seq_num = 1

        # Subscribers
        self.create_subscription(
            TicKVCache,
            "/tic_vla/kv_cache",
            self.kv_cache_callback,
            1,
        )
        self.create_subscription(
            TicImageTokens,
            "/tic_vla/image_tokens",
            self.image_tokens_callback,
            1,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        # Publishers
        self.robot_state_pub = self.create_publisher(
            Twist,
            "/tic_vla/robot_state",
            10,
        )
        self.latency_pub = self.create_publisher(
            TicLatencyMetadata,
            "/tic_vla/latency_meta",
            10,
        )
        self.action_pub = self.create_publisher(
            ActionChunk,
            "/ticvla/action_chunk",
            10,
        )

        self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)
        self.get_logger().info("[TIC-VLA Sys1] Triggering main loop...")

    def kv_cache_callback(self, msg):
        """Store latest KV cache and job metadata from sys2."""
        kv_flat = torch.tensor(list(msg.kv_values.data), dtype=torch.float32)
        seq_len = len(kv_flat) // (KV_NUM_HEADS * KV_HEAD_DIM)
        self.kv_values = (
            kv_flat.reshape(1, KV_NUM_HEADS, seq_len, KV_HEAD_DIM)
            .to(torch.bfloat16)
            .to(self.device)
        )
        self.kv_job_id = msg.job_id
        self.kv_job_stamp = msg.job_start_stamp
        self.kv_job_x = msg.job_start_x
        self.kv_job_y = msg.job_start_y
        self.kv_job_theta = msg.job_start_theta

    def image_tokens_callback(self, msg):
        """Store latest current-frame visual tokens from sys2."""
        self.image_tokens = (
            torch.tensor(list(msg.tokens.data), dtype=torch.float32)
            .reshape(1, -1, VLM_HIDDEN_SIZE)
            .to(torch.bfloat16)
            .to(self.device)
        )

    def odom_callback(self, msg):
        """Store pose and twist from odometry."""
        pose = Pose2D()
        pose.x = msg.pose.pose.position.x
        pose.y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        pose.theta = 2.0 * math.atan2(qz, qw)

        twist = Twist()
        twist.linear.x = msg.twist.twist.linear.x
        twist.linear.y = msg.twist.twist.linear.y
        twist.angular.z = msg.twist.twist.angular.z

        self.current_pose = pose
        self.current_twist = twist

    def control_loop(self):
        """Run ActionExpert forward pass and publish all outputs."""
        now_stamp = self.get_clock().now().nanoseconds / 1e9

        kv_values = self.kv_values
        image_tokens = self.image_tokens
        current_twist = self.current_twist
        job_id = self.kv_job_id
        dt, dx, dy, dtheta = self.compute_latency_metadata(now_stamp)

        # Publish robot state
        if current_twist is not None:
            self.robot_state_pub.publish(current_twist)

        latency_msg = TicLatencyMetadata()
        latency_msg.header.stamp = self.get_clock().now().to_msg()
        latency_msg.job_id = int(job_id)
        latency_msg.delta_t = float(dt)
        latency_msg.delta_x = float(dx)
        latency_msg.delta_y = float(dy)
        latency_msg.delta_theta = float(dtheta)
        self.latency_pub.publish(latency_msg)

        if kv_values is None or image_tokens is None:
            return

        # State tensor: [vx, vy, omega_z, dx, dy, dt] → (1, 6, 1)
        vx = current_twist.linear.x if current_twist else 0.0
        vy = current_twist.linear.y if current_twist else 0.0
        omega = current_twist.angular.z if current_twist else 0.0
        state = torch.tensor(
            [[[vx], [vy], [omega], [dx], [dy], [dt]]],
            dtype=torch.bfloat16,
            device=self.device,
        )  # (1, 6, 1)

        # Reconstruct kv_cache tuple — ActionExpert uses only past_key_values[-1][1]
        dummy_key = torch.empty(0, device=self.device, dtype=torch.bfloat16)
        kv_cache = ((dummy_key, kv_values),)

        with torch.inference_mode():
            waypoints = self.action_expert(image_tokens, state, kv_cache=kv_cache)

        self.publish_action_chunk(waypoints)

    def compute_latency_metadata(self, now_stamp):
        """Return (dt, dx, dy, dtheta) since the active KV cache job started.

        dx/dy are expressed in the robot frame at job start time.
        """
        if self.kv_job_stamp is None or self.current_pose is None:
            return 0.0, 0.0, 0.0, 0.0
        dt = now_stamp - self.kv_job_stamp
        dx_world = self.current_pose.x - self.kv_job_x
        dy_world = self.current_pose.y - self.kv_job_y
        cos_t = math.cos(self.kv_job_theta)
        sin_t = math.sin(self.kv_job_theta)
        dx = cos_t * dx_world + sin_t * dy_world
        dy = -sin_t * dx_world + cos_t * dy_world
        dtheta = math.atan2(
            math.sin(self.current_pose.theta - self.kv_job_theta),
            math.cos(self.current_pose.theta - self.kv_job_theta),
        )
        return dt, dx, dy, dtheta

    def publish_action_chunk(self, waypoints):
        """Convert ActionExpert output to ActionChunk and publish."""
        poses_np = waypoints[0].float().cpu().numpy()  # (T, 2)

        msg = ActionChunk()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.seq_num = self.seq_num

        for t in range(poses_np.shape[0]):
            pose = Pose2D()
            pose.x = float(poses_np[t, 0])
            pose.y = float(poses_np[t, 1])
            # Heading from direction of travel between consecutive waypoints
            if t > 0:
                ddx = poses_np[t, 0] - poses_np[t - 1, 0]
                ddy = poses_np[t, 1] - poses_np[t - 1, 1]
                pose.theta = float(math.atan2(ddy, ddx)) if (abs(ddx) + abs(ddy)) > 1e-6 else 0.0
            else:
                pose.theta = (
                    float(math.atan2(poses_np[t, 1], poses_np[t, 0]))
                    if (abs(poses_np[t, 0]) + abs(poses_np[t, 1])) > 1e-6
                    else 0.0
                )
            msg.relative_poses.append(pose)

        self.action_pub.publish(msg)
        self.seq_num += 1

    def load_action_expert(self):
        """Load only ActionExpert weights from the TIC-VLA checkpoint."""
        expert = ActionExpert(
            input_dim=VLM_HIDDEN_SIZE,
            hidden_dim=ACTION_EXPERT_HIDDEN,
            action_dim=2,
            num_layers=ACTION_EXPERT_LAYERS,
            num_chunks=ACTION_HORIZON_STEPS,
            kv_cache_feat_dim=KV_CACHE_FEAT_DIM,
        )
        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            ae_dict = {
                k.replace("model.action_expert.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.action_expert.")
            }
            expert.load_state_dict(ae_dict, strict=True)
            self.get_logger().info(f"ActionExpert weights loaded from {self.checkpoint_path}")
        else:
            self.get_logger().warn(
                f"Checkpoint not found at {self.checkpoint_path} — running with random weights"
            )
        return expert.to(torch.bfloat16).to(self.device).eval()

def main(args=None):
    rclpy.init(args=args)
    sys1_node = Sys1()
    rclpy.spin(sys1_node)
    sys1_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
