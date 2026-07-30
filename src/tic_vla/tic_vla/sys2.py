#!/usr/bin/env python3

"""TIC-VLA sys2
Two independent workloads run concurrently on separate CUDA streams:

(a) Fast encode-only loop
    Triggered by every incoming TicPixelValues message (~10 Hz).
    Runs InternViT-300M only — no LM forward pass — producing 256 visual tokens.
    Publishes to /tic_vla/image_tokens.

(b) Slow VLM reasoning loop
    Continuous back-to-back, no fixed rate (~0.5 Hz expected).
    Samples 4 frames at {t, t-3, t-6, t-9} from the rolling image buffer.
    Runs the full InternVL3 forward pass and extracts the final-layer KV values.
    Publishes to /tic_vla/kv_cache.

The two threads are required since sys1 inference depends on process (a)
"""

import math
import os
import queue
import threading
import time
from collections import deque

import torch

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from custom_msgs.msg import TicKVCache, TicImageTokens, TicPixelValues

from ticvla.models.ticvla import TICVLA


VLM_PATH             = os.path.expanduser("~/capstone/code/ticvla/InternVL3-1B")
CHECKPOINT_PATH      = os.path.expanduser("~/capstone/code/ticvla/TIC-VLA-model.ckpt")
ACTION_HORIZON_STEPS = 30
IMAGE_BUFFER_SEC     = 12.0           # covers 9s offset + margin
FRAME_OFFSETS_SEC    = [0.0, 3.0, 6.0, 9.0]
DEFAULT_INSTRUCTION  = "Go to the yelow bin."
DEVICE_TYPE          = "cuda"
IMAGE_INPUT_SIZE     = 448
KV_NUM_HEADS         = 2              # InternVL3-1B last-layer KV heads
KV_HEAD_DIM          = 64             # InternVL3-1B KV head dim
NUM_IMAGE_TOKEN      = 256            # tokens per InternViT tile


class Sys2(Node):
    def __init__(self):
        super().__init__("sys2")
        self.get_logger().info("[TIC-VLA Sys2] Initialising...")

        self.device = torch.device(DEVICE_TYPE)
        
        self.model = self.load_model()

        self.declare_parameter("instruction", DEFAULT_INSTRUCTION)
        self.instruction = self.get_parameter("instruction").get_parameter_value().string_value
        self.get_logger().info(f'Instruction: "{self.instruction}"')

        # Rolling image buffer: deque of (stamp_sec, pixel_values cpu bfloat16)
        self.buffer_lock = threading.Lock()
        self.image_buffer = deque()

        # Image encoding queue — bounded; only the latest frame matters
        self.encoding_queue = queue.Queue(maxsize=2)

        # Latest robot pose for tagging KV cache with job_start_pose
        self.pose_lock = threading.Lock()
        self.latest_pose = Pose2D()

        self.job_counter = 0

        # Separate CUDA streams so fast and slow paths don't serialise each other
        self.encoding_stream = torch.cuda.Stream()
        self.vlm_stream = torch.cuda.Stream()

        # Subscribers
        self.create_subscription(
            TicPixelValues,
            "/tic_vla/pixel_values",
            self.image_callback,
            1,
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        # Publishers
        self.tokens_pub = self.create_publisher(
            TicImageTokens,
            "/tic_vla/image_tokens",
            1,
        )
        self.kv_pub = self.create_publisher(
            TicKVCache,
            "/tic_vla/kv_cache",
            1,
        )

        threading.Thread(target=self.img_encoding_worker, daemon=True).start()
        threading.Thread(target=self.vlm_worker, daemon=True).start()

        self.get_logger().info("[TIC-VLA Sys2] Triggering main loop...")

    def image_callback(self, msg):
        """Deserialize TicPixelValues and buffer for fast encode and slow VLM paths."""
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pixel_values = (
            torch.tensor(msg.data.data, dtype=torch.float32)
            .reshape(msg.num_tiles, 3, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)
            .to(torch.bfloat16)
        )

        with self.buffer_lock:
            self.image_buffer.append((stamp, pixel_values))
            cutoff = stamp - IMAGE_BUFFER_SEC
            while self.image_buffer and self.image_buffer[0][0] < cutoff:
                self.image_buffer.popleft()

        # Drop oldest if queue is full — only the latest frame matters for fast path
        if self.encoding_queue.full():
            try:
                self.encoding_queue.get_nowait()
            except queue.Empty:
                pass
        self.encoding_queue.put_nowait((stamp, pixel_values))

    def odom_callback(self, msg):
        """Track robot pose for job_start_pose tagging on KV cache publishes."""
        pose = Pose2D()
        pose.x = msg.pose.pose.position.x
        pose.y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        pose.theta = 2.0 * math.atan2(qz, qw)
        with self.pose_lock:
            self.latest_pose = pose
    
    def img_encoding_worker(self):
        """Encode each incoming frame → 256 tokens → publish to /tic_vla/image_tokens."""
        while rclpy.ok():
            try:
                _stamp, pixel_values = self.encoding_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with torch.cuda.stream(self.encoding_stream):
                pv = pixel_values.to(self.device)
                with torch.inference_mode():
                    # (1, 3, H, W) → (num_tiles, NUM_IMAGE_TOKEN, H) → (NUM_IMAGE_TOKEN, H)
                    tokens = self.model.vlm.extract_feature(pv)
                    tokens = tokens.reshape(-1, tokens.shape[-1])
                self.encoding_stream.synchronize()

            msg = TicImageTokens()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.tokens.data = tokens.float().cpu().flatten().tolist()
            self.tokens_pub.publish(msg)

    def vlm_worker(self):
        """Continuous back-to-back full VLM pass → last-layer KV values → publish to /tic_vla/kv_cache."""
        while rclpy.ok():
            now_stamp = time.time()
            frames = self.sample_frames(now_stamp)
            if frames is None:
                time.sleep(0.1)
                continue

            with self.pose_lock:
                job_pose = (
                    self.latest_pose.x,
                    self.latest_pose.y,
                    self.latest_pose.theta,
                )
            job_id = self.job_counter
            self.job_counter += 1

            with torch.cuda.stream(self.vlm_stream):
                pixel_values, image_flags, input_ids, attention_mask = (
                    self.build_vlm_inputs(frames)
                )
                with torch.inference_mode():
                    vlm_outputs = self.model.vlm(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                        return_dict=True,
                        use_cache=True,
                    )

                past_key_values = vlm_outputs.past_key_values
                # Normalise DynamicCache to tuple of (key, value) pairs
                if hasattr(past_key_values, "layers"):
                    past_key_values = tuple(
                        (layer.keys, layer.values) for layer in past_key_values.layers
                    )

                # ActionExpert only reads past_key_values[-1][1] — send just that
                _, last_layer_values = past_key_values[-1]
                last_layer_values = last_layer_values.detach()
                # Shape: (1, KV_NUM_HEADS, seq_len, KV_HEAD_DIM)

                self.vlm_stream.synchronize()

            msg = TicKVCache()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.job_id = int(job_id)
            msg.job_start_stamp = float(now_stamp)
            msg.job_start_x = float(job_pose[0])
            msg.job_start_y = float(job_pose[1])
            msg.job_start_theta = float(job_pose[2])
            msg.kv_values.data = last_layer_values.float().cpu().flatten().tolist()
            self.kv_pub.publish(msg)

            self.get_logger().debug(
                f"Published KV cache job={job_id} shape={list(last_layer_values.shape)}"
            )

    def sample_frames(self, now_stamp):
        """Return 4 pixel_values tensors at {t, t-3, t-6, t-9} from the buffer."""
        with self.buffer_lock:
            buf = list(self.image_buffer)
        if not buf or now_stamp - buf[0][0] < FRAME_OFFSETS_SEC[-1]:
            return None
        frames = []
        for offset in FRAME_OFFSETS_SEC:
            target = now_stamp - offset
            closest = min(buf, key=lambda e: abs(e[0] - target))
            frames.append(closest[1])
        return frames

    def build_vlm_inputs(self, frames):
        """Stack frames and build tokenized prompt for InternVL3 forward pass."""
        pixel_values = torch.cat([f.to(self.device) for f in frames], dim=0)
        image_flags = torch.cat([
            torch.ones(f.shape[0], 1, dtype=torch.long, device=self.device)
            for f in frames
        ])

        system_text = (
            "You are a physical mobile robot assigned to perform navigation tasks.\n"
            "You are provided with a video consisting of visual observations, "
            "including historical and current frames.\n"
        )
        self.model.vlm.system_message = system_text
        user_text = f"The navigation instruction is: {self.instruction}"
        user_text += (
            "\nUse reasoning to predict the future target waypoints. "
            "First describe the relevant visual/navigation evidence, then return the "
            "future target waypoints for the next 3s, 6s, and 9s in format: (x, y, theta)."
        )

        # Build query string with image tokens inline
        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        generation_prompt = (
            "".join([f"Frame {i}: <image>\n" for i in range(len(frames))]) + user_text
        )
        query = generation_prompt
        for f in frames:
            image_tokens = IMG_START + (IMG_CTX * NUM_IMAGE_TOKEN * f.shape[0]) + IMG_END
            query = query.replace("<image>", image_tokens, 1)

        self.model.tokenizer.padding_side = "left"
        tokenized = self.model.tokenizer([query], return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        if getattr(self.model.vlm, "img_context_token_id", None) is None:
            self.model.vlm.img_context_token_id = (
                self.model.tokenizer.convert_tokens_to_ids(IMG_CTX)
            )

        return pixel_values, image_flags, input_ids, attention_mask

    def load_model(self):
        """Load TICVLA VLM backbone from checkpoint."""
        model = TICVLA(
            model_path=VLM_PATH,
            action_horizon_steps=ACTION_HORIZON_STEPS,
            train_vlm=False,
        )
        if os.path.exists(CHECKPOINT_PATH):
            model.load_vlm_checkpoint(CHECKPOINT_PATH)
            self.get_logger().info(f"VLM weights loaded from {CHECKPOINT_PATH}")
        else:
            self.get_logger().warn(
                f"Checkpoint not found at {CHECKPOINT_PATH} — using base InternVL3-1B weights"
            )
        return model.to(self.device).eval()


def main(args=None):
    rclpy.init(args=args)
    sys2_node = Sys2()
    rclpy.spin(sys2_node)
    sys2_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
