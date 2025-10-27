"""
Dataset utilities for SparkTrainer including splitting and validation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Stratified dataset splitter for train/val/test splits.

    Maintains class distribution across splits.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def split_manifest(self,
                      manifest_path: Union[str, Path],
                      splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                      stratify_key: Optional[str] = None,
                      output_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split manifest into train/val/test sets.

        Args:
            manifest_path: Path to manifest.jsonl
            splits: (train, val, test) ratios (must sum to 1.0)
            stratify_key: Key to stratify on (e.g., 'category', 'label')
            output_dir: Output directory for split manifests

        Returns:
            Dictionary of split_name -> samples
        """
        if abs(sum(splits) - 1.0) > 0.01:
            raise ValueError(f"Splits must sum to 1.0, got {sum(splits)}")

        # Load manifest
        samples = []
        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        logger.info(f"Loaded {len(samples)} samples from {manifest_path}")

        # Perform split
        if stratify_key:
            split_data = self._stratified_split(samples, splits, stratify_key)
        else:
            split_data = self._random_split(samples, splits)

        # Save splits if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_samples in split_data.items():
                output_path = output_dir / f"{split_name}.jsonl"
                self._save_manifest(split_samples, output_path)

        return split_data

    def _random_split(self,
                     samples: List[Dict[str, Any]],
                     splits: Tuple[float, float, float]) -> Dict[str, List[Dict[str, Any]]]:
        """Perform random split"""
        random.shuffle(samples)

        train_ratio, val_ratio, test_ratio = splits
        n = len(samples)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }

    def _stratified_split(self,
                         samples: List[Dict[str, Any]],
                         splits: Tuple[float, float, float],
                         stratify_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Perform stratified split"""
        # Group samples by stratify key
        groups = defaultdict(list)
        for sample in samples:
            # Get stratification value (handle nested keys)
            value = self._get_nested_value(sample, stratify_key)
            if value is not None:
                groups[value].append(sample)

        # Split each group
        split_data = {
            'train': [],
            'val': [],
            'test': []
        }

        for group_samples in groups.values():
            random.shuffle(group_samples)
            group_splits = self._random_split(group_samples, splits)

            for split_name, split_samples in group_splits.items():
                split_data[split_name].extend(split_samples)

        # Shuffle final splits
        for split_samples in split_data.values():
            random.shuffle(split_samples)

        logger.info(f"Stratified split: train={len(split_data['train'])}, "
                   f"val={len(split_data['val'])}, test={len(split_data['test'])}")

        return split_data

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """Get value from potentially nested dictionary"""
        keys = key.split('.')
        value = data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def _save_manifest(self, samples: List[Dict[str, Any]], output_path: Path):
        """Save samples to manifest file"""
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"Saved {len(samples)} samples to {output_path}")


class ConsistencyChecker:
    """
    Checker for dataset consistency (e.g., caption-frame matching).
    """

    def __init__(self):
        self.errors = []

    def check_manifest(self, manifest_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Check manifest for consistency issues.

        Args:
            manifest_path: Path to manifest.jsonl

        Returns:
            Dictionary of check results
        """
        issues = []
        stats = {
            'total_samples': 0,
            'missing_files': 0,
            'empty_captions': 0,
            'invalid_metadata': 0
        }

        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    sample = json.loads(line)
                    stats['total_samples'] += 1

                    # Check file existence
                    if 'frames_dir' in sample:
                        frames_dir = Path(sample['frames_dir'])
                        if not frames_dir.exists():
                            issues.append({
                                'line': line_num,
                                'type': 'missing_frames_dir',
                                'message': f"Frames directory not found: {frames_dir}"
                            })
                            stats['missing_files'] += 1

                    if 'audio' in sample:
                        audio_path = Path(sample['audio'])
                        if not audio_path.exists():
                            issues.append({
                                'line': line_num,
                                'type': 'missing_audio',
                                'message': f"Audio file not found: {audio_path}"
                            })
                            stats['missing_files'] += 1

                    # Check captions
                    if 'meta' in sample:
                        meta = sample['meta']
                        if 'captions' in meta:
                            captions = meta['captions']
                            if not captions or all(not c.strip() for c in captions):
                                issues.append({
                                    'line': line_num,
                                    'type': 'empty_caption',
                                    'message': 'Caption is empty'
                                })
                                stats['empty_captions'] += 1

                except json.JSONDecodeError:
                    issues.append({
                        'line': line_num,
                        'type': 'json_error',
                        'message': 'Failed to parse JSON'
                    })
                    stats['invalid_metadata'] += 1

        result = {
            'valid': len(issues) == 0,
            'stats': stats,
            'issues': issues
        }

        logger.info(f"Consistency check: {stats['total_samples']} samples, "
                   f"{len(issues)} issues found")

        return result

    def check_caption_quality(self,
                            manifest_path: Union[str, Path],
                            min_caption_length: int = 5,
                            max_caption_length: int = 200) -> Dict[str, Any]:
        """
        Check caption quality.

        Args:
            manifest_path: Path to manifest
            min_caption_length: Minimum caption length
            max_caption_length: Maximum caption length

        Returns:
            Quality check results
        """
        issues = []
        stats = {
            'total_captions': 0,
            'too_short': 0,
            'too_long': 0,
            'avg_length': 0.0
        }

        caption_lengths = []

        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    sample = json.loads(line)

                    if 'meta' in sample and 'captions' in sample['meta']:
                        for caption in sample['meta']['captions']:
                            length = len(caption.strip())
                            caption_lengths.append(length)
                            stats['total_captions'] += 1

                            if length < min_caption_length:
                                issues.append({
                                    'line': line_num,
                                    'type': 'caption_too_short',
                                    'message': f'Caption too short: {length} chars',
                                    'caption': caption
                                })
                                stats['too_short'] += 1

                            elif length > max_caption_length:
                                issues.append({
                                    'line': line_num,
                                    'type': 'caption_too_long',
                                    'message': f'Caption too long: {length} chars',
                                    'caption': caption[:50] + '...'
                                })
                                stats['too_long'] += 1

                except:
                    pass

        if caption_lengths:
            stats['avg_length'] = sum(caption_lengths) / len(caption_lengths)

        return {
            'valid': len(issues) == 0,
            'stats': stats,
            'issues': issues
        }


def split_dataset(manifest_path: Union[str, Path],
                 output_dir: Union[str, Path],
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 stratify_key: Optional[str] = None,
                 seed: int = 42) -> Dict[str, str]:
    """
    Convenience function to split dataset.

    Args:
        manifest_path: Path to manifest
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify_key: Key to stratify on
        seed: Random seed

    Returns:
        Dictionary of split_name -> output_path
    """
    splitter = DatasetSplitter(seed=seed)
    splitter.split_manifest(
        manifest_path=manifest_path,
        splits=(train_ratio, val_ratio, test_ratio),
        stratify_key=stratify_key,
        output_dir=output_dir
    )

    output_dir = Path(output_dir)
    return {
        'train': str(output_dir / 'train.jsonl'),
        'val': str(output_dir / 'val.jsonl'),
        'test': str(output_dir / 'test.jsonl')
    }


def check_dataset_consistency(manifest_path: Union[str, Path]) -> bool:
    """
    Check dataset consistency.

    Args:
        manifest_path: Path to manifest

    Returns:
        True if consistent, False otherwise
    """
    checker = ConsistencyChecker()
    result = checker.check_manifest(manifest_path)
    return result['valid']
