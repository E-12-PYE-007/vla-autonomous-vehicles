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


ACTION_HORIZON_STEPS = 30
IMAGE_BUFFER_SEC     = 12.0           # covers 9s offset + margin
FRAME_OFFSETS_SEC    = [0.0, 3.0, 6.0, 9.0]
DEVICE_TYPE          = "cuda"
IMAGE_INPUT_SIZE     = 448
KV_NUM_HEADS         = 2              # InternVL3-1B last-layer KV heads
KV_HEAD_DIM          = 64             # InternVL3-1B KV head dim
NUM_IMAGE_TOKEN      = 256            # tokens per InternViT tile
# Matches upstream. Not a useful tuning dial: the model self-terminates around 120-190
# tokens, so 128 and 200 cost the same ~5s and both let the thought finish. Only a cap
# low enough to truncate (tried 64) saves time, and that dropped waypoint magnitudes ~5x
# with the robot pinned to MIN_FORWARD_SPEED. VLM latency has to come from elsewhere.
MAX_NEW_TOKENS       = 200
POSE_HISTORY_SEC     = 12.0           # pose buffer span for previous-waypoint text
MAX_PREV_WAYPOINTS   = 9              # training caps history at 90 steps @10 Hz = 9 s


class Sys2(Node):
    def __init__(self):
        super().__init__("sys2")
        self.get_logger().info("[TIC-VLA Sys2] Initialising...")

        self.device = torch.device(DEVICE_TYPE)

        self.declare_parameter("vlm_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.vlm_path = self.get_parameter("vlm_path").get_parameter_value().string_value
        self.checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value

        self.model = self.load_model()

        self.declare_parameter("instruction", "")
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
        self.pose_history = deque()     # (stamp, x, y, theta), trimmed to POSE_HISTORY_SEC
        self.start_stamp = None         # wall time of first odom, anchors elapsed_time

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

        # Buffered against wall-clock arrival, not msg.header.stamp: in sim the header
        # carries Gazebo simulation time, which starts at 0 and is not comparable to the
        # time.time() the sampler works in.
        wall = time.time()
        with self.buffer_lock:
            self.image_buffer.append((wall, pixel_values, msg.img_seq_num))
            cutoff = wall - IMAGE_BUFFER_SEC
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
        """Track robot pose for job_start_pose tagging and previous-waypoint history."""
        pose = Pose2D()
        pose.x = msg.pose.pose.position.x
        pose.y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        pose.theta = 2.0 * math.atan2(qz, qw)
        stamp = time.time()
        with self.pose_lock:
            self.latest_pose = pose
            if self.start_stamp is None:
                self.start_stamp = stamp
            self.pose_history.append((stamp, pose.x, pose.y, pose.theta))
            cutoff = stamp - POSE_HISTORY_SEC
            while self.pose_history and self.pose_history[0][0] < cutoff:
                self.pose_history.popleft()

    def format_previous_waypoints(self):
        """Describe recent motion as 1 Hz displacements, matching the training prompt.

        Training feeds the VLM its own recent trajectory here. Without it the model has no
        record of what it has done and invents one ("I have reached the chair and begun
        reversing"), which then drives the reasoning and the waypoints.
        """
        with self.pose_lock:
            hist = list(self.pose_history)
            start_stamp = self.start_stamp

        now = time.time()
        elapsed = 0.0 if start_stamp is None else now - start_stamp

        waypoints = []
        if hist:
            # Walk back in whole seconds, taking the displacement over each second in the
            # robot frame as it was at the start of that second.
            for k in range(MAX_PREV_WAYPOINTS, 0, -1):
                t_end = now - (k - 1) * 1.0
                t_start = now - k * 1.0
                if t_start < hist[0][0]:
                    continue
                s = min(hist, key=lambda e: abs(e[0] - t_start))
                e = min(hist, key=lambda e: abs(e[0] - t_end))
                dx_w = e[1] - s[1]
                dy_w = e[2] - s[2]
                cos_t, sin_t = math.cos(s[3]), math.sin(s[3])
                x_rel = cos_t * dx_w + sin_t * dy_w
                y_rel = -sin_t * dx_w + cos_t * dy_w
                # The benchmark driver skips all-zero waypoints, so a stationary robot
                # reports "No waypoints available" rather than a run of (0.00, 0.00, 0.00)
                # — which the model answered with fabricated arrival stories.
                if abs(x_rel) < 1e-6 and abs(y_rel) < 1e-6:
                    continue
                waypoints.append(f"({x_rel:.2f}, {y_rel:.2f}, 0.00)")

        if waypoints:
            return (
                f"From 0.0s to current timestamp time is {elapsed:.1f}s. "
                f"(a list of waypoints 1s in between): {', '.join(waypoints)}\n"
                "Each waypoint (x, y, z) is the displacement over the previous 1.0s. "
                "x is forward, y is left, z is up."
            )
        return f"From 0.0s to current timestamp time is {elapsed:.1f}s. No waypoints available."
    
    def img_encoding_worker(self):
        """Encode each incoming frame → 256 tokens → publish to /tic_vla/image_tokens."""
        while rclpy.ok():
            try:
                _stamp, pixel_values = self.encoding_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            with torch.cuda.stream(self.encoding_stream):
                # The ActionExpert expects one whole-frame tile (NUM_IMAGE_TOKEN tokens).
                # image_processing already publishes a single tile; guard against a
                # multi-tile publisher by taking the thumbnail, which dynamic_preprocess
                # appends last whenever it splits.
                pv = pixel_values[-1:] if pixel_values.shape[0] > 1 else pixel_values
                pv = pv.to(self.device)
                with torch.inference_mode():
                    # (1, 3, H, W) → (1, NUM_IMAGE_TOKEN, H) → (NUM_IMAGE_TOKEN, H)
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
                # reasoning is already baked into input_ids as the assistant turn; it is
                # returned only for inspection, so it is not consumed here.
                pixel_values, image_flags, input_ids, attention_mask, _reasoning = (
                    self.build_vlm_inputs(frames)
                )
                with torch.no_grad():
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

            self.get_logger().info(
                f"Published KV cache job={job_id} shape={list(last_layer_values.shape)}"
            )

    def sample_frames(self, now_stamp):
        """Return the pixel_values at {t-9, t-6, t-3, t}, oldest first.

        Mirrors DynaNav's _get_sampled_image_paths: take only the offsets the history can
        actually satisfy, so the VLM gets 1 frame at startup growing to 4 as history
        accrues, then drop duplicates so a still image is never presented as a history.

        Matching is on wall-clock arrival. Comparing time.time() against the sim-time
        header stamp made every offset resolve to the newest frame, so the VLM saw four
        copies of one instant and the 9 s warm-up guard never bit.

        Order matters: training builds the frame list by walking history forward and
        appending the current frame last, so the model reads Frame 0 -> Frame N as time
        moving forward. Feeding it newest-first reverses apparent motion and the
        ActionExpert predicts backwards waypoints.
        """
        with self.buffer_lock:
            buf = list(self.image_buffer)
        if not buf:
            return None

        span = now_stamp - buf[0][0]
        offsets = [o for o in FRAME_OFFSETS_SEC if o <= span] or [0.0]

        entries = []
        for offset in sorted(offsets, reverse=True):
            target = now_stamp - offset
            entries.append(min(buf, key=lambda e: abs(e[0] - target)))

        seen, frames = set(), []
        for entry in entries:
            if entry[2] not in seen:
                seen.add(entry[2])
                frames.append(entry[1])
        return frames

    def build_vlm_inputs(self, frames):
        """Generate the reasoning turn, then tokenize the full conversation for the
        forward pass that produces the KV cache.

        TIC-VLA conditions the ActionExpert on the VLM's *reasoning*, which lives in the
        KV cache. So the cache has to come from a conversation that already contains the
        assistant's answer — generating first, then running the forward pass over
        system + user + assistant, exactly as upstream inference does.
        """
        pixel_values = torch.cat([f.to(self.device) for f in frames], dim=0)
        num_patches_list = [f.shape[0] for f in frames]
        image_flags = torch.ones(
            pixel_values.shape[0], 1, dtype=torch.long, device=self.device
        )

        # "wheeled robot" matches the benchmark driver (robot_type="wheeled robot"), which
        # is also the robot_type tag in the DynaNav training data.
        system_text = (
            "You are a wheeled robot assigned to perform navigation tasks.\n"
            "You are provided with a video consisting of visual observations, "
            "including historical and current frames.\n"
        )
        self.model.vlm.system_message = system_text
        # Wording matches generate_and_extract_kv_cache in DynaNav/ticvla.py (the
        # benchmark path drops the "First describe the evidence" sentence used by the
        # sync demo).
        user_text = f"The navigation instruction is: {self.instruction}"
        user_text += f"\n{self.format_previous_waypoints()}"
        user_text += (
            "\nUse reasoning to predict future target waypoints for the next 3s, 6s, "
            "and 9s in format: (x, y, theta). Each waypoint represents the cumulative "
            "offset from the current position (total displacement over 3s, 6s, or 9s), "
            "where x is positive for forward, y is positive for left, and theta is the "
            "heading angle in radians."
        )

        generation_prompt = (
            "".join([f"Frame {i}: <image>\n" for i in range(len(frames))]) + user_text
        )

        # 1) Generate the reasoning response.
        # Matches DynaNav/ticvla.py predict_async — the deployed benchmark path — not the
        # temperature-0.7 sync demo in ticvla/models/ticvla.py. Near-greedy sampling keeps
        # scene grounding consistent while retaining a little diversity (pure greedy locked
        # onto "I have reached the chair"; 0.7 re-rolled the scene every cycle).
        generation_config = dict(
            max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.1, top_p=0.1, top_k=10
        )
        with torch.no_grad():
            generated_response = self.model.vlm.chat(
                self.model.tokenizer,
                pixel_values,
                generation_prompt,
                generation_config,
                history=None,
                return_history=False,
                num_patches_list=num_patches_list,
            )

        # 2) Rebuild the full conversation, including that response.
        messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [{"type": "image", "image": ""} for _ in frames]
                + [{"type": "text", "text": user_text}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": generated_response}]},
        ]
        text_batch = self.model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if isinstance(text_batch, list):
            text_batch = text_batch[0] if text_batch else ""

        # 3) Expand each <image> placeholder into IMG_CONTEXT tokens for its tile count.
        IMG_START, IMG_END, IMG_CTX = "<img>", "</img>", "<IMG_CONTEXT>"
        query = text_batch
        if "<image>" in query:
            for tiles in num_patches_list:
                if tiles > 0:
                    query = query.replace(
                        "<image>", IMG_START + (IMG_CTX * NUM_IMAGE_TOKEN * tiles) + IMG_END, 1
                    )
        else:
            prepend = [
                IMG_START + (IMG_CTX * NUM_IMAGE_TOKEN * tiles) + IMG_END
                for tiles in num_patches_list
                if tiles > 0
            ]
            if prepend:
                query = "\n".join(prepend) + "\n" + query

        self.model.tokenizer.padding_side = "left"
        tokenized = self.model.tokenizer([query], return_tensors="pt", padding=True)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        if getattr(self.model.vlm, "img_context_token_id", None) is None:
            self.model.vlm.img_context_token_id = (
                self.model.tokenizer.convert_tokens_to_ids(IMG_CTX)
            )

        return pixel_values, image_flags, input_ids, attention_mask, generated_response

    def load_model(self):
        """Load TICVLA VLM backbone from checkpoint."""
        model = TICVLA(
            model_path=self.vlm_path,
            action_horizon_steps=ACTION_HORIZON_STEPS,
            train_vlm=False,
        )
        if os.path.exists(self.checkpoint_path):
            model.load_vlm_checkpoint(self.checkpoint_path)
            self.get_logger().info(f"VLM weights loaded from {self.checkpoint_path}")
        else:
            self.get_logger().warn(
                f"Checkpoint not found at {self.checkpoint_path} — using base InternVL3-1B weights"
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
