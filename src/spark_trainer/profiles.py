"""
Profile and persistent settings management for SparkTrainer.

This module provides functionality for:
- Loading and saving reusable configuration profiles
- Managing persistent user settings
- Environment configuration
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class UserSettings:
    """Persistent user settings stored in ~/.spark_trainer/config.json"""

    # UI Preferences
    theme: str = "dark"
    language: str = "en"
    default_view: str = "dashboard"

    # Paths
    default_dataset_dir: Optional[str] = None
    default_output_dir: Optional[str] = None
    default_model_cache: Optional[str] = None

    # Compute preferences
    default_gpu_partition: Optional[str] = None
    default_precision: str = "fp16"
    enable_deepspeed: bool = False

    # Notifications
    enable_notifications: bool = True
    notify_on_completion: bool = True
    notify_on_error: bool = True

    # Recent activity
    recent_datasets: List[str] = None
    recent_models: List[str] = None
    recent_experiments: List[str] = None

    # Advanced
    max_parallel_jobs: int = 4
    log_level: str = "INFO"
    telemetry_enabled: bool = True

    def __post_init__(self):
        if self.recent_datasets is None:
            self.recent_datasets = []
        if self.recent_models is None:
            self.recent_models = []
        if self.recent_experiments is None:
            self.recent_experiments = []


class ProfileManager:
    """Manages configuration profiles and persistent settings"""

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the profile manager.

        Args:
            base_dir: Base directory for SparkTrainer (defaults to ~/.spark_trainer)
        """
        if base_dir is None:
            self.base_dir = Path.home() / ".spark_trainer"
        else:
            self.base_dir = Path(base_dir)

        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.settings_file = self.base_dir / "config.json"
        self.profiles_file = self.base_dir / "profiles.yaml"

        # Initialize settings and profiles
        self._settings: Optional[UserSettings] = None
        self._profiles: Dict[str, Any] = {}

        # Load existing settings and profiles
        self.load_settings()
        self.load_profiles()

    # Settings Management

    def load_settings(self) -> UserSettings:
        """Load persistent settings from config.json"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                self._settings = UserSettings(**data)
            except Exception as e:
                print(f"Warning: Failed to load settings: {e}. Using defaults.")
                self._settings = UserSettings()
        else:
            self._settings = UserSettings()
            self.save_settings()

        return self._settings

    def save_settings(self) -> None:
        """Save settings to config.json"""
        with open(self.settings_file, 'w') as f:
            json.dump(asdict(self._settings), f, indent=2)

    def get_settings(self) -> UserSettings:
        """Get current settings"""
        if self._settings is None:
            self.load_settings()
        return self._settings

    def update_settings(self, **kwargs) -> UserSettings:
        """
        Update settings with provided key-value pairs.

        Example:
            manager.update_settings(theme="light", log_level="DEBUG")
        """
        settings = self.get_settings()
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self.save_settings()
        return settings

    def add_recent_dataset(self, dataset_id: str, max_recent: int = 10) -> None:
        """Add a dataset to recent activity"""
        settings = self.get_settings()
        if dataset_id in settings.recent_datasets:
            settings.recent_datasets.remove(dataset_id)
        settings.recent_datasets.insert(0, dataset_id)
        settings.recent_datasets = settings.recent_datasets[:max_recent]
        self.save_settings()

    def add_recent_model(self, model_id: str, max_recent: int = 10) -> None:
        """Add a model to recent activity"""
        settings = self.get_settings()
        if model_id in settings.recent_models:
            settings.recent_models.remove(model_id)
        settings.recent_models.insert(0, model_id)
        settings.recent_models = settings.recent_models[:max_recent]
        self.save_settings()

    def add_recent_experiment(self, exp_id: str, max_recent: int = 10) -> None:
        """Add an experiment to recent activity"""
        settings = self.get_settings()
        if exp_id in settings.recent_experiments:
            settings.recent_experiments.remove(exp_id)
        settings.recent_experiments.insert(0, exp_id)
        settings.recent_experiments = settings.recent_experiments[:max_recent]
        self.save_settings()

    # Profile Management

    def load_profiles(self) -> Dict[str, Any]:
        """Load configuration profiles from profiles.yaml"""
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, 'r') as f:
                    self._profiles = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load profiles: {e}. Using empty profiles.")
                self._profiles = {}
        else:
            # Create default profiles
            self._profiles = self._create_default_profiles()
            self.save_profiles()

        return self._profiles

    def save_profiles(self) -> None:
        """Save profiles to profiles.yaml"""
        with open(self.profiles_file, 'w') as f:
            yaml.dump(self._profiles, f, default_flow_style=False, sort_keys=False)

    def get_profiles(self) -> Dict[str, Any]:
        """Get all profiles"""
        return self._profiles

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific profile by name"""
        return self._profiles.get(name)

    def create_profile(self, name: str, config: Dict[str, Any],
                       description: str = "", tags: List[str] = None) -> None:
        """
        Create a new configuration profile.

        Args:
            name: Profile name
            config: Configuration dictionary
            description: Profile description
            tags: List of tags for categorization
        """
        self._profiles[name] = {
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "config": config
        }
        self.save_profiles()

    def update_profile(self, name: str, config: Dict[str, Any]) -> None:
        """Update an existing profile"""
        if name in self._profiles:
            self._profiles[name]["config"].update(config)
            self._profiles[name]["updated_at"] = datetime.now().isoformat()
            self.save_profiles()

    def delete_profile(self, name: str) -> bool:
        """Delete a profile"""
        if name in self._profiles:
            del self._profiles[name]
            self.save_profiles()
            return True
        return False

    def list_profiles(self, tag: Optional[str] = None) -> List[str]:
        """
        List all profile names, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of profile names
        """
        if tag:
            return [
                name for name, profile in self._profiles.items()
                if tag in profile.get("tags", [])
            ]
        return list(self._profiles.keys())

    def _create_default_profiles(self) -> Dict[str, Any]:
        """Create default configuration profiles"""
        return {
            "quick-vision-language": {
                "description": "Fast vision-language fine-tuning for small datasets",
                "tags": ["vision-language", "quick", "finetune"],
                "created_at": datetime.now().isoformat(),
                "config": {
                    "model_name": "Salesforce/blip2-opt-2.7b",
                    "model_type": "vision_language",
                    "batch_size": 8,
                    "learning_rate": 5e-5,
                    "num_epochs": 3,
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "mixed_precision": "fp16",
                    "use_accelerate": True,
                    "gradient_accumulation_steps": 2,
                    "warmup_steps": 100
                }
            },
            "production-vision-language": {
                "description": "Production-grade vision-language training with DeepSpeed",
                "tags": ["vision-language", "production", "deepspeed"],
                "created_at": datetime.now().isoformat(),
                "config": {
                    "model_name": "Salesforce/blip2-opt-6.7b",
                    "model_type": "vision_language",
                    "batch_size": 4,
                    "learning_rate": 1e-5,
                    "num_epochs": 10,
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "mixed_precision": "bf16",
                    "use_accelerate": True,
                    "use_deepspeed": True,
                    "deepspeed_config": "configs/deepspeed_zero3.json",
                    "gradient_accumulation_steps": 4,
                    "gradient_checkpointing": True,
                    "warmup_steps": 500,
                    "eval_steps": 1000,
                    "save_steps": 1000
                }
            },
            "video-diffusion-standard": {
                "description": "Standard video diffusion model training",
                "tags": ["diffusion", "video", "standard"],
                "created_at": datetime.now().isoformat(),
                "config": {
                    "model_name": "stabilityai/stable-video-diffusion-img2vid",
                    "model_type": "diffusion",
                    "batch_size": 1,
                    "learning_rate": 1e-5,
                    "num_epochs": 100,
                    "optimizer": "adamw",
                    "mixed_precision": "fp16",
                    "gradient_accumulation_steps": 8,
                    "gradient_checkpointing": True,
                    "save_steps": 5000
                }
            },
            "whisper-asr-finetune": {
                "description": "Whisper ASR fine-tuning for custom audio",
                "tags": ["asr", "audio", "whisper"],
                "created_at": datetime.now().isoformat(),
                "config": {
                    "model_name": "openai/whisper-small",
                    "model_type": "asr",
                    "batch_size": 16,
                    "learning_rate": 1e-5,
                    "num_epochs": 5,
                    "optimizer": "adamw",
                    "scheduler": "linear",
                    "mixed_precision": "fp16",
                    "gradient_accumulation_steps": 2,
                    "warmup_steps": 500
                }
            },
            "multimodal-apotheon": {
                "description": "Unified multimodal training (text+image+video+audio)",
                "tags": ["multimodal", "apotheon", "advanced"],
                "created_at": datetime.now().isoformat(),
                "config": {
                    "model_type": "multimodal",
                    "vision_encoder": "openai/clip-vit-large-patch14",
                    "audio_encoder": "openai/whisper-small",
                    "text_model": "meta-llama/Llama-3.2-3B",
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                    "num_epochs": 20,
                    "optimizer": "adamw",
                    "scheduler": "cosine",
                    "mixed_precision": "bf16",
                    "use_deepspeed": True,
                    "deepspeed_config": "configs/deepspeed_zero3.json",
                    "gradient_accumulation_steps": 4,
                    "gradient_checkpointing": True,
                    "curriculum_learning": True,
                    "multi_task_weights": {
                        "captioning": 0.3,
                        "transcription": 0.3,
                        "contrastive": 0.4
                    }
                }
            },
            "preprocessing-comprehensive": {
                "description": "Comprehensive video preprocessing with all features",
                "tags": ["preprocessing", "video", "comprehensive"],
                "created_at": datetime.now().isoformat(),
                "config": {
                    "extract_frames": True,
                    "frame_rate": 1.0,
                    "resolution": "224x224",
                    "extract_audio": True,
                    "transcribe": True,
                    "whisper_model": "openai/whisper-small",
                    "generate_captions": True,
                    "captioner_backend": "blip2",
                    "detect_scenes": True,
                    "scene_threshold": 30.0,
                    "smart_sampling": True,
                    "motion_threshold": 0.3,
                    "perceptual_hashing": True,
                    "num_workers": 4,
                    "deterministic_layout": True
                }
            }
        }

    # Environment Information

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get comprehensive environment information.

        Returns:
            Dictionary with system, CUDA, PyTorch, and package info
        """
        import platform
        import sys

        env_info = {
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.machine(),
            },
            "spark_trainer": {
                "base_dir": str(self.base_dir),
                "settings_file": str(self.settings_file),
                "profiles_file": str(self.profiles_file),
            }
        }

        # CUDA/PyTorch info
        try:
            import torch
            env_info["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }

            if torch.cuda.is_available():
                env_info["gpus"] = []
                for i in range(torch.cuda.device_count()):
                    gpu_info = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                        "total_memory": torch.cuda.get_device_properties(i).total_memory,
                    }
                    env_info["gpus"].append(gpu_info)
        except ImportError:
            env_info["pytorch"] = {"error": "PyTorch not installed"}

        # Transformers info
        try:
            import transformers
            env_info["transformers"] = {
                "version": transformers.__version__
            }
        except ImportError:
            env_info["transformers"] = {"error": "Transformers not installed"}

        # FFmpeg info
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                env_info["ffmpeg"] = {
                    "available": True,
                    "version": version_line
                }
            else:
                env_info["ffmpeg"] = {"available": False}
        except:
            env_info["ffmpeg"] = {"available": False}

        return env_info


# Global instance
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager(base_dir: Optional[Path] = None) -> ProfileManager:
    """Get or create the global ProfileManager instance"""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager(base_dir)
    return _profile_manager
