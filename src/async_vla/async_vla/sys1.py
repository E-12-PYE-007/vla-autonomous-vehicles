#!/usr/bin/env python3

import importlib.util
import os
import sys
import types
from collections import deque
from functools import lru_cache

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image as PILImage

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from custom_msgs.msg import ActionChunk, AsyncHiddenState, ImageWithSeqNum


def _load_edge_adapter():
    # A direct `from prismatic.models.small_head import Edge_adapter` initialises
    # prismatic/__init__.py, which chains through prismatic/vla/datasets and imports
    # tensorflow — a package with no aarch64 wheel. This function loads small_head.py
    # via importlib and pre-registers stub modules for the prismatic.vla.constants
    # names it needs, so the full package init is never executed.
    asyncvla_source = next(
        (p for p in sys.path if os.path.exists(os.path.join(p, 'prismatic'))),
        None,
    )
    if asyncvla_source is None:
        raise RuntimeError(
            'AsyncVLA source not found on sys.path. '
            'Ensure the AsyncVLA repo root is prepended to sys.path before '
            'importing this module (sys1_dsk / sys1_rcp do this automatically).'
        )

    constants = types.ModuleType('prismatic.vla.constants')
    constants.ACTION_DIM = 4
    constants.ACTION_TOKEN_BEGIN_IDX = 0
    constants.IGNORE_INDEX = -100
    constants.NUM_ACTIONS_CHUNK = 8
    constants.STOP_INDEX = 2

    prismatic_pkg = types.ModuleType('prismatic')
    prismatic_pkg.__path__ = [os.path.join(asyncvla_source, 'prismatic')]
    vla_pkg = types.ModuleType('prismatic.vla')
    vla_pkg.__path__ = [os.path.join(asyncvla_source, 'prismatic', 'vla')]

    sys.modules.setdefault('prismatic', prismatic_pkg)
    sys.modules.setdefault('prismatic.vla', vla_pkg)
    sys.modules['prismatic.vla.constants'] = constants

    # The public visualnav-transformer submodule may not define MultiLayerDecoder_idcat
    # (an older name). Patch it so small_head.py can import without error.
    try:
        from vint_train.models.vint import self_attention
        if not hasattr(self_attention, 'MultiLayerDecoder_idcat'):
            self_attention.MultiLayerDecoder_idcat = self_attention.MultiLayerDecoder_trans
    except Exception:
        pass

    small_head_path = os.path.join(asyncvla_source, 'prismatic', 'models', 'small_head.py')
    spec = importlib.util.spec_from_file_location('asyncvla_small_head_runtime', small_head_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Edge_adapter


Edge_adapter = _load_edge_adapter()


RESUME_STEP = 750000
OBS_ENCODING_SIZE = 1024
MHA_NUM_ATTENTION_HEADS = 4
MHA_NUM_ATTENTION_LAYERS = 4
MHA_FF_DIM_FACTOR = 4
IMAGE_INTERMEDIATE_SIZE = (224, 224)  # PIL resize before tensor resize (matches reference)
IMAGE_SIZE = (96, 96)
METRIC_WAYPOINT_SPACING = 0.1 
IMAGE_BUFFER_SIZE = 80  # 8 secs worth of data
SYS1_RATE_HZ = 8.0


_normalise = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class Sys1(Node):
    def __init__(self):
        super().__init__("sys1")
        self.get_logger().info("[AsyncVLA Sys1] initialising...")

        self.declare_parameter("shead_path", "")
        shead_path = self.get_parameter("shead_path").get_parameter_value().string_value

        # Load model
        shead, self.device = _load_model(shead_path, RESUME_STEP)
        self.inference = Inference(shead, self.device)
        self.get_logger().info("[AsyncVLA Sys1] Model loaded")

        # Initialise states
        self.bridge = CvBridge()
        self.curr_img = None
        self.curr_img_seq_num = None
        self.img_buffer = {}
        self.img_buffer_keys = deque(maxlen=IMAGE_BUFFER_SIZE)
        self.latest_hidden_state = None
        self.latest_hidden_seq_num = None
        # Carried through from sys2 for logging only.
        self.latest_hidden_end_seq_num = None
        self.latest_sys2_inference_ms = 0.0

        # Publishers
        self.action_chunk_pub = self.create_publisher(ActionChunk, "/asyncvla/action_chunk", 1)

        # Subscribers
        self.create_subscription(AsyncHiddenState, "/asyncvla/hidden_state", self.hidden_state_callback, 1)
        self.create_subscription(ImageWithSeqNum, "/cam", self.img_callback, 1)

        self.create_timer(1.0 / SYS1_RATE_HZ, self.timer_callback)
        self.get_logger().info("[AsyncVLA Sys1] Triggering main control loop...")

    def img_callback(self, msg: ImageWithSeqNum):
        img = PILImage.fromarray(self.bridge.compressed_imgmsg_to_cv2(msg.img, desired_encoding="rgb8"))
        processed = _process_image(img, self.device)

        # Evict oldest entry before it gets dropped from the deque
        if len(self.img_buffer_keys) == IMAGE_BUFFER_SIZE:
            self.img_buffer.pop(self.img_buffer_keys[0], None)
        self.img_buffer_keys.append(msg.img_seq_num)
        self.img_buffer[msg.img_seq_num] = processed
        self.curr_img = processed
        self.curr_img_seq_num = msg.img_seq_num

    def hidden_state_callback(self, msg: AsyncHiddenState):
        self.latest_hidden_state = torch.tensor(msg.hidden_states.data, dtype=torch.float32).reshape(1, 8, 1024).to(torch.bfloat16).to(self.device)
        self.latest_hidden_seq_num = msg.img_seq_num
        self.latest_hidden_end_seq_num = msg.end_img_seq_num
        self.latest_sys2_inference_ms = msg.inference_ms

    def timer_callback(self):
        projected_actions = self.latest_hidden_state
        seq_num = self.latest_hidden_seq_num
        past_img = self.img_buffer.get(seq_num)
        curr_img = self.curr_img
        # Captured alongside curr_img so the two stay paired even if a new frame
        # arrives between now and publish_action_chunk.
        curr_img_seq_num = self.curr_img_seq_num
        # Same reason: keep these with the hidden state they describe.
        end_img_seq_num = self.latest_hidden_end_seq_num
        sys2_inference_ms = self.latest_sys2_inference_ms

        if projected_actions is None or curr_img is None or past_img is None:
            return

        poses = self.inference.run(curr_img, past_img, projected_actions)
        self.publish_action_chunk(poses, seq_num, curr_img_seq_num, end_img_seq_num, sys2_inference_ms)

    def publish_action_chunk(
        self,
        poses: np.ndarray,
        img_seq_num: int,
        curr_img_seq_num: int,
        end_img_seq_num: int,
        sys2_inference_ms: float,
    ):
        chunk = ActionChunk()
        chunk.header.stamp = self.get_clock().now().to_msg()
        chunk.seq_num = img_seq_num
        chunk.curr_img_seq_num = curr_img_seq_num
        chunk.end_img_seq_num = end_img_seq_num
        chunk.sys2_inference_ms = sys2_inference_ms

        for t in range(poses.shape[1]):
            pose = Pose2D()
            # y is passed through unmirrored: the model already emits ROS convention
            # (+y = left). Measured in unempty_office_square against known object
            # poses, with all four objects inside the camera's +/-31deg FOV:
            #     chair   raw y +0.820  true bearing +19.6deg (left)
            #     desk    raw y +0.266  true bearing +16.0deg (left)
            #     box     raw y -0.746  true bearing  -8.1deg (right)
            #     cabinet raw y -1.828  true bearing -25.7deg (right)
            # Rank order and sign both match, and the resulting bearings land within a
            # few degrees for the box and cabinet. Upstream run_action_head applies
            # `dy = -dy` inside its own pd_controller, which then feeds a robot with the
            # opposite steering sign; re-applying it here mirrored every chunk and made
            # the robot drive to the object opposite the one it was asked for.
            # theta is left as the model emits it: upstream never steers from
            # per-waypoint heading, so there is no reference for its sign. Consumers
            # should prefer the positions.
            pose.x = float(poses[0, t, 0]) * METRIC_WAYPOINT_SPACING
            pose.y = float(poses[0, t, 1]) * METRIC_WAYPOINT_SPACING
            pose.theta = float(np.arctan2(poses[0, t, 3], poses[0, t, 2]))
            chunk.relative_poses.append(pose)

        self.action_chunk_pub.publish(chunk)
        self.get_logger().info(f"[AsyncVLA Sys1] Published action chunk seq={img_seq_num}")


class Inference:
    def __init__(self, shead, device):
        self.shead = shead
        self.device = device

    def run(self, curr_img: torch.Tensor, past_img: torch.Tensor, projected_actions: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            predicted_delta = self.shead(curr_img, past_img, projected_actions)
            predicted_poses = _delta_to_pose(predicted_delta)
        return predicted_poses.cpu().float().numpy()


@lru_cache(maxsize=1)
def _load_model(shead_path: str, resume_step: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    shead = Edge_adapter(
        obs_encoding_size=OBS_ENCODING_SIZE,
        mha_num_attention_heads=MHA_NUM_ATTENTION_HEADS,
        mha_num_attention_layers=MHA_NUM_ATTENTION_LAYERS,
        mha_ff_dim_factor=MHA_FF_DIM_FACTOR,
    )

    checkpoint_path = os.path.join(shead_path, f"shead--{resume_step}_checkpoint.pt")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    shead.load_state_dict(state_dict, strict=False)
    shead.to(torch.bfloat16).to(device)
    # Edge_adapter contains dropout layers; without eval() they stay active and the same
    # input produces a different trajectory every call. The noise was as large as the
    # effect of changing the hidden state, so the language instruction had almost no
    # influence on the output. Upstream run_vla.py does the same (.eval() on vla and
    # action_proj).
    shead.eval()

    return shead, device


def _process_image(img: PILImage.Image, device: torch.device) -> torch.Tensor:
    img = img.resize(IMAGE_INTERMEDIATE_SIZE, PILImage.BILINEAR)
    tensor = TF.to_tensor(img)
    tensor = TF.resize(tensor, list(IMAGE_SIZE)).unsqueeze(0)
    return _normalise(tensor).to(device).to(torch.bfloat16)


def _delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """Convert predicted delta actions [N, T, 4] to absolute poses [N, T, 4]."""
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    N, T = dx.shape
    poses = []
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    for t in range(1, T):
        ct, st = torch.cos(theta), torch.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))

    return torch.stack(poses, dim=1)


def main(args=None):
    rclpy.init(args=args)

    sys1_node = Sys1()
    rclpy.spin(sys1_node)

    sys1_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
