import sys, os
sys.path.insert(0, '..')

import rclpy
from rclpy.node import Node
from custom_msgs.msg import ActionChunk, ImageWithSeqNum
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D

from functools import lru_cache
import time, math
from typing import Optional, Tuple, Type


import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# ---------------------------
# Custom Imports
# ---------------------------
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

# Language-only is always modality 7 (see header).
MODALITY_ID_LANGUAGE_ONLY = 7
INFERENCE_RATE = 3.0 #Hz
RESUME_STEP = 210000 # Checkpoint step to load model from
METRIC_WAYPOINT_SPACING = 0.1 #Obtained from VMAX/Inference rate -- same convention as ViNT

# ===============================================================
# Utility Functions
# ===============================================================
def clip_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]. (The original script referenced this without defining it.)"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

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
) -> nn.Module:
    module = module_class(**module_args)
    count_parameters(module, module_name)

    state_dict = load_checkpoint(module_name, vla_path, resume_step)
    module.load_state_dict(state_dict)

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
    ).to(device) #            trust_remote_code=True,
    
    vla.vision_backbone.set_num_images_in_input(2) # Required even when not using goal image for correct tensor shape
    vla.to(dtype=torch.bfloat16, device=device)

    # TODO: Verify if this needs to be passed or initialised
    pose_projector = init_module(
        vla_path,
        RESUME_STEP,
        ProprioProjector,
        "pose_projector",
        device,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},            
    )
    
    action_head = init_module(
        vla_path,
        RESUME_STEP,
        L1RegressionActionHead_idcat,
        "action_head",
        device,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": ACTION_DIM},            
        to_bf16=True,
    )            
 
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

        # Load Model (???)
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
        self.get_logger().info("[AsyncVLA Sys1] Triggering main control loop...")

    def img_callback(self, msg: ImageWithSeqNum):
        img = PILImage.fromarray(self.bridge.imgmsg_to_cv2(msg.img, desired_encoding="rgb8"))
        seq_num = msg.img_seq_num
        self.latest_img = [img, seq_num]

    def inference_timer_callback(self):
        if self.latest_img.img is None:
            self.get_logger().info(f"[OmniVLA] No image ready for inference")
            return
        actions = self.inference.run(self.latest_img.img, self.goal_text)
        self.publish_action_chunk(actions, self.latest_img.seq_num)

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
        self.get_logger().info(f"[AsyncVLA Sys1] Published action chunk seq={img_seq_num}")


# ===============================================================
# Inference Class
# ===============================================================
class Inference:
    def __init__(self, vla, action_head, pose_projector, device, num_patches, action_tokenizer, processor):
        self.vla = vla
        self.action_head = action_head
        self.pose_projector = pose_projector #TODO: Check if required?
        self.device = device
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.processor = processor
        self.count_id = 0
        #TODO: Check if all of these are needed.
    
    def run(self, img_PIL, goal_text):
        batch = self.data_transformer_omnivla(
            img_PIL,
            goal_text,
            goal_image_PIL=img_PIL, # duplicate the current img to maintain tensor shape. It is masked out later.
            goal_pose_loc_norm=np.zeros(POSE_DIM, dtype=np.float32),
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
        )

        # Run forward pass
        actions = self.run_forward_pass(
            vla=self.vla.eval(),
            action_head=self.action_head.eval(),
            pose_projector=self.pose_projector.eval(),
            batch=batch,
            device_id=self.device,
            num_patches=self.NUM_PATCHES,
        )

        return actions

    # ----------------------------
    # Custom Collator
    # ----------------------------
    def collator_custom(self, instances, model_max_length, pad_token_id, pixel_values_dtype=torch.float32):
        IGNORE_INDEX = -100
        input_ids = pad_sequence([inst["input_ids"] for inst in instances], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([inst["labels"] for inst in instances], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids, labels = input_ids[:, :model_max_length], labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        # Stack current + (dummy) goal image on the channel dim -> (B, 2C, H, W)
        pixel_values_current = [inst["pixel_values_current"] for inst in instances]
        pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
        pixel_values = torch.cat((torch.stack(pixel_values_current), torch.stack(pixel_values_goal)), dim=1)

        actions = torch.stack([torch.from_numpy(np.copy(inst["actions"])) for inst in instances])
        goal_pose = torch.stack([torch.from_numpy(np.copy(inst["goal_pose"])) for inst in instances])

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )

    # ----------------------------
    # Transform Data to Dataset Format
    # ----------------------------
    def transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                           current_image_PIL, goal_image_PIL, action_tokenizer,
                           base_tokenizer, image_transform, predict_stop_token=True):
        IGNORE_INDEX = -100
        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]

        prompt_builder = PurePromptBuilder("openvla")
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = torch.tensor(base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids)
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            actions=torch.as_tensor(actions),
            goal_pose=goal_pose_cos_sin,
        )

    # ----------------------------
    # Data Transformer for OmniVLA
    # ----------------------------
    def data_transformer_omnivla(self, current_image_PIL, lan_inst, goal_image_PIL, goal_pose_loc_norm,
                                 action_tokenizer, processor):
        actions = np.random.rand(8, 4)  # dummy actions -- only used to shape the prompt's action tokens

        batch_data = self.transform_datatype(
            lan_inst, actions, goal_pose_loc_norm,
            current_image_PIL, goal_image_PIL,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
        )

        batch = self.collator_custom(
            instances=[batch_data],
            model_max_length=processor.tokenizer.model_max_length,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        return batch

    # ----------------------------
    # Run Forward Pass
    # ----------------------------
    def run_forward_pass(self, vla, action_head, pose_projector, batch, device_id, num_patches) -> torch.Tensor:
        modality_id = torch.as_tensor([MODALITY_ID_LANGUAGE_ONLY], dtype=torch.float32)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        # Prepare data for action extraction
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        # Get last-layer hidden states for the action portion of the response
        last_hidden_states = output.hidden_states[-1]  # (B, seq_len, D)
        text_hidden_states = last_hidden_states[:, num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )  # (B, act_chunk_len, D)

        with torch.no_grad():
            predicted_actions = action_head.predict_action(
                actions_hidden_states, modality_id.to(torch.bfloat16).to(device_id)
            )

        return predicted_actions
    


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

