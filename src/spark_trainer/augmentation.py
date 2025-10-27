"""
Augmentation management system for SparkTrainer.

Loads and applies augmentation pipelines from augmentations.yaml.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class AugmentationManager:
    """Manages data augmentation pipelines"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize augmentation manager.

        Args:
            config_path: Path to augmentations.yaml (None = use default)
        """
        if config_path is None:
            # Default to configs/augmentations.yaml
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "configs" / "augmentations.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load augmentation configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded augmentation config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load augmentation config: {e}")
            return {}

    def get_pipeline(self, pipeline_name: str) -> Dict[str, List[str]]:
        """
        Get augmentation pipeline by name.

        Args:
            pipeline_name: Pipeline name (e.g., 'vision_language')

        Returns:
            Dictionary of modality -> augmentation groups
        """
        pipelines = self.config.get('pipelines', {})
        return pipelines.get(pipeline_name, {})

    def get_augmentations(self, modality: str, groups: List[str]) -> List[Dict[str, Any]]:
        """
        Get augmentations for a modality and groups.

        Args:
            modality: Modality (e.g., 'image', 'video', 'audio')
            groups: List of augmentation groups (e.g., ['basic', 'color'])

        Returns:
            List of augmentation configurations
        """
        modality_config = self.config.get(modality, {})
        augmentations = []

        for group in groups:
            group_augs = modality_config.get(group, [])
            augmentations.extend(group_augs)

        return augmentations

    def list_pipelines(self) -> List[str]:
        """List available pipeline names"""
        return list(self.config.get('pipelines', {}).keys())

    def list_modalities(self) -> List[str]:
        """List available modalities"""
        modalities = []
        for key in self.config.keys():
            if key not in ['pipelines', 'settings']:
                modalities.append(key)
        return modalities


# Global instance
_augmentation_manager: Optional[AugmentationManager] = None


def get_augmentation_manager() -> AugmentationManager:
    """Get or create global augmentation manager"""
    global _augmentation_manager
    if _augmentation_manager is None:
        _augmentation_manager = AugmentationManager()
    return _augmentation_manager
