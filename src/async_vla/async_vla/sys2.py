#!/usr/bin/env python3

from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout

from custom_msgs.msg import AsyncHiddenState, ImageWithSeqNum

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.small_head import Proj_Actiontokens
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast


VLA_PATH = './AsyncVLA_release'
RESUME_STEP = 750000
DEFAULT_GOAL = 'Go to the yellow bin'

class Sys2(Node):
    def __init__(self):
        super().__init__("sys2")

        # Load model
        vla, action_proj, device, num_patches, action_tokenizer, processor = _load_model(VLA_PATH, RESUME_STEP)
        self.inference = Inference(vla, action_proj, device, num_patches, action_tokenizer, processor)
        self.get_logger().info('[AsyncVLA Sys2] Model loaded')

        # Prompt for goal — press Enter for default
        user_input = input('Enter goal: ').strip()
        self.goal_text = user_input if user_input else DEFAULT_GOAL
        self.get_logger().info(f"[AsyncVLA Sys2] Goal set as: '{self.goal_text}'")

        # Publishers
        self.hidden_state_pub = self.create_publisher(
            AsyncHiddenState, 
            '/asyncvla/hidden_state', 
            1
        )

        # Subscribers
        self.bridge = CvBridge()
        self.create_subscription(
            ImageWithSeqNum, 
            '/cam', 
            self.img_callback, 
            1
        )

    def img_callback(self, msg: ImageWithSeqNum):
        img = Image.fromarray(self._bridge.imgmsg_to_cv2(msg.current_image, desired_encoding='rgb8'))

        actions = self.inference.run(img, self.goal_text)

        hidden_state_msg = AsyncHiddenState()
        hidden_state_msg.header.stamp = self.get_clock().now().to_msg()
        hidden_state_msg.img_seq_num = msg.img_seq_num

        hidden_state_msg.hidden_states.data = actions.reshape(-1).astype(np.float32).tolist()

        self.hidden_state_pub.publish(hidden_state_msg)
        self.get_logger().info(f'[AsyncVLA Sys2] Published hidden state for img_seq={hidden_state_msg.img_seq_num}')

class Inference:
    def __init__(self, vla, action_proj, device, num_patches, action_tokenizer, processor):
        self.vla = vla
        self.action_proj = action_proj
        self.device = device
        self.num_patches = num_patches
        self.action_tokenizer = action_tokenizer
        self.processor = processor

    def run(self, img: Image.Image, goal_text: str) -> np.ndarray:
        batch = self._prepare_batch(img, goal_text)
        return self._forward(batch)

    def _prepare_batch(self, img: Image.Image, goal_text: str) -> dict:
        IGNORE_INDEX = -100
        actions = np.random.rand(8, 4)  # dummy actions for token layout only

        action_chunk_string = (
            self.action_tokenizer(actions[0])
            + ''.join(self.action_tokenizer(actions[1:]))
        )

        prompt_builder = PurePromptBuilder('openvla')
        prompt_builder.add_turn('human', f'What action should the robot take to {goal_text}?')
        prompt_builder.add_turn('gpt', action_chunk_string)

        tokenizer = self.processor.tokenizer
        input_ids = torch.tensor(
            tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[:-(len(action_chunk_string) + 1)] = IGNORE_INDEX

        pixel_values = self.processor.image_processor.apply_transform(img)

        input_ids_b = pad_sequence([input_ids], batch_first=True, padding_value=tokenizer.pad_token_id)
        labels_b = pad_sequence([labels], batch_first=True, padding_value=IGNORE_INDEX)
        input_ids_b = input_ids_b[:, :tokenizer.model_max_length]
        labels_b = labels_b[:, :tokenizer.model_max_length]

        stacked = torch.stack([pixel_values])
        pixel_values_b = torch.cat([stacked, stacked], dim=1)  # duplicate as goal image

        return {
            'pixel_values': pixel_values_b,
            'input_ids': input_ids_b,
            'attention_mask': input_ids_b.ne(tokenizer.pad_token_id),
            'labels': labels_b,
        }

    def _forward(self, batch: dict) -> np.ndarray:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        modality_id = torch.as_tensor([7], dtype=torch.float32, device=self.device)

        with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                pixel_values=batch['pixel_values'].to(torch.bfloat16).to(self.device),
                modality_id=modality_id.to(torch.bfloat16),
                labels=batch['labels'].to(self.device),
                output_hidden_states=True,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=False,
            )

        gt_token_ids = batch['labels'][:, 1:].to(self.device)
        action_mask = get_current_action_mask(gt_token_ids) | get_next_actions_mask(gt_token_ids)

        text_hidden = output.hidden_states[-1][:, self.num_patches:-1]
        batch_size = batch['input_ids'].shape[0]
        actions_hidden = (
            text_hidden[action_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            projected = self.action_proj.predict_action(
                actions_hidden.detach(), modality_id.to(torch.bfloat16)
            )

        return projected.detach().to(torch.float32).cpu().numpy()

def _remove_ddp_prefix(state_dict: dict) -> dict:
    return {k.removeprefix('module.'): v for k, v in state_dict.items()}

def _load_checkpoint(module_name: str, path: str, step: int) -> dict:
    import os
    checkpoint_path = os.path.join(path, f'{module_name}--{step}_checkpoint.pt')
    if not os.path.exists(checkpoint_path) and module_name == 'pose_projector':
        checkpoint_path = os.path.join(path, f'proprio_projector--{step}_checkpoint.pt')
    return _remove_ddp_prefix(torch.load(checkpoint_path, map_location='cpu'))

@lru_cache(maxsize=1)
def _load_model(vla_path: str, resume_step: int):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    AutoConfig.register('openvla', OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.to(dtype=torch.bfloat16, device=device)

    action_proj = Proj_Actiontokens(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=1024
    )
    action_proj.load_state_dict(_load_checkpoint('action_proj', vla_path, resume_step))
    action_proj = action_proj.to(torch.bfloat16).to(device)

    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return vla, action_proj, device, num_patches, action_tokenizer, processor

def main(args=None):
    rclpy.init(args=args)

    sys2_node = Sys2()
    rclpy.spin(sys2_node)

    sys2_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
