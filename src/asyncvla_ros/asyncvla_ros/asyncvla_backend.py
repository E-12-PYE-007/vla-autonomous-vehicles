from __future__ import annotations

# Dataclass is used for a simple structured prediction output
from dataclasses import dataclass

# Typing helpers for cleaner type hints
from typing import List, Optional, Tuple

# Used to temporarily change the working directory safely
from contextlib import contextmanager

# Standard Python libraries
import os
import sys

# NumPy for array handling
import numpy as np

# PIL for saving/loading images in the format expected by AsyncVLA
from PIL import Image


@dataclass
class AsyncVLAPrediction:
    """
    Simple container for the model output.
    Stores a list of predicted relative poses as:
    (x, y, theta)
    """
    relative_poses: List[Tuple[float, float, float]]


class AsyncVLABackend:
    """
    ROS-facing wrapper around the official AsyncVLA inference script.

    This backend acts as a bridge between your ROS nodes and the public
    AsyncVLA codebase. It prepares inputs, loads the model, runs inference,
    and converts the result into a cleaner ROS-friendly format.
    """

    def __init__(
        self,
        model_repo_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        # Path to the AsyncVLA repository root
        self.model_repo_path = model_repo_path

        # Path to the trained checkpoint directory
        self.checkpoint_path = checkpoint_path

        # Device to run inference on, usually "cuda" or "cpu"
        self.device = device

        # Flag to track whether the model has been loaded
        self.is_loaded = False

        # Official AsyncVLA objects that will be initialized later
        self.run_asyncvla_module = None
        self.vla = None
        self.action_head = None
        self.pose_projector = None
        self.shead = None
        self.action_proj = None
        self.device_id = None
        self.NUM_PATCHES = None
        self.action_tokenizer = None
        self.processor = None
        self.inference = None

    @contextmanager
    def _pushd(self, path: str):
        """
        Temporarily change the current working directory.
        This is needed because the AsyncVLA repo relies on relative paths.
        """
        old_cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def _validate_paths(self) -> None:
        """
        Check that the model repo path and checkpoint path exist.
        Raise clear errors if they are missing or invalid.
        """
        if not self.model_repo_path:
            raise ValueError("model_repo_path is required")
        if not os.path.isdir(self.model_repo_path):
            raise FileNotFoundError(f"model_repo_path does not exist: {self.model_repo_path}")

        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        if not os.path.isdir(self.checkpoint_path):
            raise FileNotFoundError(f"checkpoint_path does not exist: {self.checkpoint_path}")

    def _ensure_inference_dir(self) -> str:
        """
        Ensure the AsyncVLA inference folder exists.
        This is where input images are written before inference.
        """
        inference_dir = os.path.join(self.model_repo_path, "inference")
        os.makedirs(inference_dir, exist_ok=True)
        return inference_dir

    def _save_numpy_rgb_to_png(self, image_np: np.ndarray, path: str) -> None:
        """
        Save a NumPy image array to PNG.

        Ensures the image is uint8 and has shape HxWx3.
        """
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {image_np.shape}")

        Image.fromarray(image_np).save(path)

    def _goal_pose_to_loc_norm(
        self,
        goal_pose: Optional[Tuple[float, float, float]],
        metric_waypoint_spacing: float,
    ) -> np.ndarray:
        """
        Convert a robot-frame goal pose (x, y, yaw) into the format expected
        by AsyncVLA:

            [x_norm, y_norm, cos(yaw), sin(yaw)]

        The x and y values are normalized by metric_waypoint_spacing.
        """
        if goal_pose is None:
            # Default "no pose goal" representation
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        x, y, yaw = goal_pose
        return np.array(
            [
                float(x) / metric_waypoint_spacing,
                float(y) / metric_waypoint_spacing,
                np.cos(float(yaw)),
                np.sin(float(yaw)),
            ],
            dtype=np.float32,
        )

    def _set_modality_flags(
        self,
        goal_text: Optional[str],
        goal_pose: Optional[Tuple[float, float, float]],
        goal_image: Optional[np.ndarray],
    ) -> None:
        """
        Set the global modality flags used by the public AsyncVLA script.

        These flags tell the script which type of goal is being used:
        - language prompt
        - pose goal
        - image goal
        """
        if self.run_asyncvla_module is None:
            raise RuntimeError("AsyncVLA module not imported")

        self.run_asyncvla_module.satellite = False
        self.run_asyncvla_module.lan_prompt = bool(goal_text and goal_text.strip())
        self.run_asyncvla_module.pose_goal = goal_pose is not None
        self.run_asyncvla_module.image_goal = goal_image is not None

    def load_model(self) -> None:
        """
        Load the official AsyncVLA model and related components.
        This must be called before predict().
        """
        # Make sure input paths are valid
        self._validate_paths()

        # Add the AsyncVLA repo to Python import path if needed
        if self.model_repo_path not in sys.path:
            sys.path.insert(0, self.model_repo_path)

        # Add visualnav-transformer repo to Python import path if needed
        visualnav_path = os.path.expanduser("~/models/visualnav-transformer")
        if visualnav_path not in sys.path:
            sys.path.insert(0, visualnav_path)

        # Run imports from the AsyncVLA repo root because the official code
        # depends on relative paths such as ./config_nav/... and ./inference/...
        with self._pushd(self.model_repo_path):
            from inference import run_asyncvla as run_asyncvla_module

            # Save reference to the imported module
            self.run_asyncvla_module = run_asyncvla_module

            # Extract required classes/functions from the module
            InferenceConfig = run_asyncvla_module.InferenceConfig
            define_model = run_asyncvla_module.define_model
            Inference = run_asyncvla_module.Inference

            # Create default inference config
            cfg = InferenceConfig()

            # Override checkpoint path with the user-provided one
            cfg.vla_path = self.checkpoint_path

            # CPU support is only a soft hint for now
            if self.device.lower() == "cpu":
                pass

            # Build/load model components
            (
                self.vla,
                self.action_head,
                self.pose_projector,
                self.shead,
                self.action_proj,
                self.device_id,
                self.NUM_PATCHES,
                self.action_tokenizer,
                self.processor,
            ) = define_model(cfg)

            # Create a blank placeholder goal image because the official
            # inference pipeline expects one to always exist
            blank_goal = Image.new("RGB", (224, 224), color=(0, 0, 0))

            # Create inference helper object
            self.inference = Inference(
                save_dir=os.path.join(self.model_repo_path, "inference"),
                lan_inst_prompt="",
                goal_utm=(0.0, 0.0),
                goal_compass=0.0,
                goal_image_PIL=blank_goal,
                action_tokenizer=self.action_tokenizer,
                processor=self.processor,
            )

        self.is_loaded = True
        print("AsyncVLA model loaded successfully")

    def preprocess_inputs(
        self,
        past_image: np.ndarray,
        current_image: np.ndarray,
        goal_pose: Optional[Tuple[float, float, float]] = None,
        goal_text: Optional[str] = None,
        goal_image: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Prepare model inputs in a consistent dictionary format.
        """
        if past_image is None or current_image is None:
            raise ValueError("Both past_image and current_image are required")

        # Fixed spacing used to normalize waypoint coordinates
        metric_waypoint_spacing = 0.1

        # Convert pose goal to AsyncVLA's expected normalized format
        goal_pose_loc_norm = self._goal_pose_to_loc_norm(
            goal_pose=goal_pose,
            metric_waypoint_spacing=metric_waypoint_spacing,
        )

        # Return all prepared inputs
        return {
            "past_image": past_image,
            "current_image": current_image,
            "goal_pose": goal_pose,
            "goal_text": goal_text if goal_text is not None else "",
            "goal_image": goal_image,
            "goal_pose_loc_norm": goal_pose_loc_norm,
            "metric_waypoint_spacing": metric_waypoint_spacing,
        }

    def _write_asyncvla_input_files(
        self,
        past_image: np.ndarray,
        current_image: np.ndarray,
        goal_image: Optional[np.ndarray] = None,
    ) -> None:
        """
        Write input images to the exact file paths expected by the public
        AsyncVLA script.

        Files written:
        - ./inference/past.png
        - ./inference/cur.png
        - ./inference/goal.png
        """
        inference_dir = self._ensure_inference_dir()

        past_path = os.path.join(inference_dir, "past.png")
        cur_path = os.path.join(inference_dir, "cur.png")
        goal_path = os.path.join(inference_dir, "goal.png")

        # Save previous and current camera images
        self._save_numpy_rgb_to_png(past_image, past_path)
        self._save_numpy_rgb_to_png(current_image, cur_path)

        # Save goal image, or create a blank placeholder if none is provided
        if goal_image is None:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            self._save_numpy_rgb_to_png(blank, goal_path)
        else:
            self._save_numpy_rgb_to_png(goal_image, goal_path)

    def postprocess_to_relative_poses(self, actions_list) -> List[Tuple[float, float, float]]:
        """
        Convert AsyncVLA output into a list of relative poses (x, y, theta).

        Expected AsyncVLA output format is approximately:
            [x, y, cos(theta), sin(theta)]
        """
        if actions_list is None or len(actions_list) == 0:
            return []

        # Use the final trajectory in the list, which should be the most current one
        traj = actions_list[-1]

        # Convert PyTorch tensor to NumPy if needed
        if hasattr(traj, "detach"):
            traj = traj.detach().cpu().numpy()

        # Validate expected shape: [1, T, 4]
        if traj.ndim != 3 or traj.shape[0] < 1 or traj.shape[2] < 4:
            raise ValueError(f"Unexpected AsyncVLA trajectory shape: {traj.shape}")

        relative_poses: List[Tuple[float, float, float]] = []

        # Convert each predicted step into (x, y, theta)
        for t in range(traj.shape[1]):
            x = float(traj[0, t, 0])
            y = float(traj[0, t, 1])

            cos_theta = float(traj[0, t, 2])
            sin_theta = float(traj[0, t, 3])
            theta = float(np.arctan2(sin_theta, cos_theta))

            relative_poses.append((x, y, theta))

        return relative_poses

    def predict(
        self,
        past_image: np.ndarray,
        current_image: np.ndarray,
        goal_pose: Optional[Tuple[float, float, float]] = None,
        goal_text: Optional[str] = None,
        goal_image: Optional[np.ndarray] = None,
    ) -> AsyncVLAPrediction:
        """
        Run full AsyncVLA inference:
        1. preprocess inputs
        2. write image files
        3. set modality flags
        4. run official inference
        5. convert output to relative poses
        """
        if not self.is_loaded:
            raise RuntimeError("AsyncVLABackend used before load_model() was called")

        # Prepare all model inputs
        model_inputs = self.preprocess_inputs(
            past_image=past_image,
            current_image=current_image,
            goal_pose=goal_pose,
            goal_text=goal_text,
            goal_image=goal_image,
        )

        # Write images to disk in the format/location expected by AsyncVLA
        self._write_asyncvla_input_files(
            past_image=model_inputs["past_image"],
            current_image=model_inputs["current_image"],
            goal_image=model_inputs["goal_image"],
        )

        # Set goal modality flags in the imported AsyncVLA module
        self._set_modality_flags(
            goal_text=model_inputs["goal_text"],
            goal_pose=model_inputs["goal_pose"],
            goal_image=model_inputs["goal_image"],
        )

        # Run inference from the repo root so relative paths work correctly
        with self._pushd(self.model_repo_path):
            # Update goal image used by the inference object
            goal_img_path = os.path.join(self.model_repo_path, "inference", "goal.png")
            self.inference.goal_image_PIL = Image.open(goal_img_path).convert("RGB").resize(
                (224, 224), Image.BILINEAR
            )

            # Update language prompt on the inference object
            self.inference.lan_inst_prompt = model_inputs["goal_text"]

            # Call the official AsyncVLA forward pass
            actions_list, modality_id = self.inference.run_forward_pass(
                vla=self.vla.eval(),
                action_head=self.action_head.eval(),
                action_proj=self.action_proj.eval(),
                shead=self.shead.eval(),
                noisy_action_projector=None,
                pose_projector=self.pose_projector.eval(),
                current_image_PIL=Image.fromarray(
                    model_inputs["current_image"].astype(np.uint8)
                ).convert("RGB").resize((224, 224), Image.BILINEAR),
                lan_inst=model_inputs["goal_text"],
                goal_pose_loc_norm=model_inputs["goal_pose_loc_norm"],
                metric_waypoint_spacing=model_inputs["metric_waypoint_spacing"],
                action_tokenizer=self.action_tokenizer,
                device_id=self.device_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_film=False,
                num_patches=self.NUM_PATCHES,
                compute_diffusion_l1=False,
                num_diffusion_steps_train=None,
                mode="train",
                idrun=0,
            )

        # Convert raw model output into relative poses
        relative_poses = self.postprocess_to_relative_poses(actions_list)

        # Return structured prediction object
        return AsyncVLAPrediction(relative_poses=relative_poses)