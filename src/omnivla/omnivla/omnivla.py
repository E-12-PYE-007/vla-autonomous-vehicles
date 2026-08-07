import os
from functools import lru_cache
from typing import Type

import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import rclpy
from rclpy.node import Node
from custom_msgs.msg import ActionChunk, ImageWithSeqNum
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM

from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor

# Language-only is always modality 7
MODALITY_ID_LANGUAGE_ONLY = 7
INFERENCE_RATE = 3.0 #Hz
RESUME_STEP = 210000 # Checkpoint step to load model from
METRIC_WAYPOINT_SPACING = 0.1 #Obtained from VMAX/Inference rate -- same convention as ViNT

# ===============================================================
# Utility Functions
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)

def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")

def init_module(
    vla_path,
    resume_step,
    module_class: Type[nn.Module],
    module_name: str,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False
) -> nn.Module:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    state_dict = load_checkpoint(module_name, vla_path, resume_step)
    module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return module

# ===============================================================
# Utility Functions - Ours
# ===============================================================
@lru_cache(maxsize=1)
def _load_model(vla_path: str, resume_step: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    # Load processor and VLA
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    
    vla.vision_backbone.set_num_images_in_input(2) # Required even when not using goal image for correct tensor shape
    vla.to(dtype=torch.bfloat16, device=device)

    # Required even though not using pose goal to maintain tensor shape for mask
    pose_projector = init_module(
        vla_path,
        resume_step,
        ProprioProjector,
        "pose_projector",
        device,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
    )
    
    action_head = init_module(
        vla_path,
        resume_step,
        L1RegressionActionHead_idcat,
        "action_head",
        device,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
        to_bf16=True,
    )

    # Sets models to evaluation mode - i.e. no dropout
    vla.eval()
    action_head.eval()
    pose_projector.eval()     
 
    # Get number of vision patches
    NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()    
    NUM_PATCHES += 1 #for goal pose

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_head, pose_projector, device, NUM_PATCHES, action_tokenizer, processor




# ===============================================================
# Node Class
# ===============================================================
class Omnivla(Node):
    def __init__(self):
        super().__init__("Omnivla")
        self.get_logger().info("[OmniVLA] initialising...")

        # Harness to allow different model storage location per machine (rcp vs desktop).
        #   Set by launch file
        self.declare_parameter("vla_path", "")
        vla_path = self.get_parameter("vla_path").get_parameter_value().string_value

        # Load model
        vla, action_head, pose_projector, device, NUM_PATCHES, action_tokenizer, processor = _load_model(vla_path, RESUME_STEP)
        self.inference = Inference(
            vla,
            action_head,
            pose_projector,
            device,
            NUM_PATCHES,
            action_tokenizer=action_tokenizer,
            processor=processor,
    )
        self.get_logger().info("[OmniVLA] Model loaded")


        self.declare_parameter("goal", "")
        self.goal_text = self.get_parameter("goal").get_parameter_value().string_value
        self.get_logger().info(f"[OmniVLA] Goal set as: '{self.goal_text}'")

        # Publishers
        self.action_chunk_pub = self.create_publisher(ActionChunk, "/asyncvla/action_chunk", 1)

        # Subscribers
        self.bridge = CvBridge()
        self.create_subscription(ImageWithSeqNum, "/cam", self.img_callback, 1)
        self.latest_img = {
                            "img": None,
                            "seq_num": None
                        }
        # Inference timer
        self.create_timer(1.0 / INFERENCE_RATE, self.inference_timer_callback)
        self.get_logger().info("[OmniVLA] Triggering main control loop...")

    def img_callback(self, msg: ImageWithSeqNum):
        img = PILImage.fromarray(self.bridge.imgmsg_to_cv2(msg.img, desired_encoding="rgb8"))
        seq_num = msg.img_seq_num
        self.latest_img["img"] = img
        self.latest_img["seq_num"] = seq_num

    def inference_timer_callback(self):
        if self.latest_img["img"] is None:
            self.get_logger().info("[OmniVLA] No image ready for inference")
            return
        actions = self.inference.run(self.latest_img["img"], self.goal_text)
        self.publish_action_chunk(actions, self.latest_img["seq_num"])

    def publish_action_chunk(self, poses: np.ndarray, img_seq_num: int):
        chunk = ActionChunk()
        chunk.header.stamp = self.get_clock().now().to_msg()
        chunk.seq_num = img_seq_num

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
        self.get_logger().info(f"[OmniVLA] Published action chunk seq={img_seq_num}")


# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, vla, action_head, pose_projector, device, num_patches, action_tokenizer, processor):
        self.vla = vla
        self.action_head = action_head
        self.pose_projector = pose_projector  # TODO: Check if required?
        self.device = device
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.processor = processor

    def run(self, img: PILImage.Image, goal_text: str) -> np.ndarray:
        batch = self._prepare_batch(img, goal_text)
        return self._forward(batch)

    def _prepare_batch(self, img: PILImage.Image, goal_text: str) -> dict:
        IGNORE_INDEX = -100
        actions = np.random.rand(8, 4)  # dummy actions for token layout only

        action_chunk_string = self.action_tokenizer(actions[0]) + "".join(self.action_tokenizer(actions[1:]))

        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {goal_text}?")
        prompt_builder.add_turn("gpt", action_chunk_string)

        tokenizer = self.processor.tokenizer
        input_ids = torch.tensor(tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[: -(len(action_chunk_string) + 1)] = IGNORE_INDEX

        pixel_values = self.processor.image_processor.apply_transform(img)
        stacked = torch.stack([pixel_values])
        pixel_values_b = torch.cat([stacked, stacked], dim=1)  # duplicate as goal image (masked out for lang-only)

        input_ids_b = pad_sequence([input_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels_b = pad_sequence([labels], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids_b = input_ids_b[:, : tokenizer.model_max_length]
        labels_b = labels_b[:, : tokenizer.model_max_length]

        return {
            "pixel_values": pixel_values_b,
            "input_ids": input_ids_b,
            "attention_mask": input_ids_b.ne(tokenizer.pad_token_id),
            "labels": labels_b,
            "goal_pose": torch.zeros(1, POSE_DIM, dtype=torch.float32),
        }

    def _forward(self, batch: dict) -> np.ndarray:
        modality_id = torch.as_tensor([MODALITY_ID_LANGUAGE_ONLY], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                modality_id=modality_id.to(torch.bfloat16).to(self.device),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self.device),
                proprio_projector=self.pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        gt_token_ids = batch["labels"][:, 1:].to(self.device)
        action_mask = get_current_action_mask(gt_token_ids) | get_next_actions_mask(gt_token_ids)

        text_hidden_states = output.hidden_states[-1][:, self.num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[action_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted_actions = self.action_head.predict_action(
                actions_hidden_states, modality_id.to(torch.bfloat16).to(self.device)
            )

        return predicted_actions.detach().to(torch.float32).cpu().numpy()
    


# ===============================================================
# Main Body
# ===============================================================
def main(args=None):
    rclpy.init(args=args)

    omnivla_node = Omnivla()
    rclpy.spin(omnivla_node)

    omnivla_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

