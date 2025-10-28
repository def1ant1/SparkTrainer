"""
Auto-generate Dataset Cards following HuggingFace schema.

Includes:
- Dataset metadata
- Provenance and lineage
- License information
- Curation steps
- Statistics and quality metrics
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import Counter
import yaml

logger = logging.getLogger(__name__)


class DatasetCardGenerator:
    """
    Auto-generate dataset cards in HuggingFace format.

    Example structure:
    ---
    dataset_info:
      features:
        - name: image
          dtype: image
        - name: text
          dtype: string
      splits:
        - name: train
          num_examples: 1000
        - name: val
          num_examples: 200
      download_size: 512000000
      dataset_size: 600000000
    ---

    # Dataset Card for MyDataset

    ## Dataset Description
    ...
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        description: Optional[str] = None,
        license: Optional[str] = None,
        citation: Optional[str] = None,
        homepage: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.description = description or f"Dataset: {dataset_name}"
        self.license = license
        self.citation = citation
        self.homepage = homepage
        self.tags = tags or []

        self.metadata = {}
        self.statistics = {}
        self.curation_steps = []

    def analyze_manifest(
        self,
        manifest_path: str,
        split_name: str = "train",
    ):
        """
        Analyze dataset manifest and extract statistics.

        Args:
            manifest_path: Path to JSONL manifest
            split_name: Split name (train, val, test)
        """
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise ValueError(f"Manifest not found: {manifest_path}")

        items = []
        with open(manifest_path, 'r') as f:
            for line in f:
                items.append(json.loads(line.strip()))

        # Extract statistics
        num_examples = len(items)
        content_types = Counter(item.get('content_type', 'unknown') for item in items)
        mime_types = Counter(item.get('mime_type', 'unknown') for item in items)

        # Calculate total size
        total_size = sum(item.get('size_bytes', 0) for item in items)

        # Content-specific stats
        if 'image' in content_types:
            image_items = [item for item in items if item.get('content_type') == 'image']
            widths = [item.get('metadata', {}).get('width', 0) for item in image_items]
            heights = [item.get('metadata', {}).get('height', 0) for item in image_items]

            self.statistics['images'] = {
                'count': len(image_items),
                'avg_width': sum(widths) / len(widths) if widths else 0,
                'avg_height': sum(heights) / len(heights) if heights else 0,
                'formats': Counter(item.get('metadata', {}).get('format', 'unknown') for item in image_items),
            }

        if 'video' in content_types:
            video_items = [item for item in items if item.get('content_type') == 'video']
            durations = [item.get('metadata', {}).get('duration', 0) for item in video_items]
            fps_values = [item.get('metadata', {}).get('fps', 0) for item in video_items]

            self.statistics['videos'] = {
                'count': len(video_items),
                'total_duration': sum(durations),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'avg_fps': sum(fps_values) / len(fps_values) if fps_values else 0,
            }

        if 'audio' in content_types:
            audio_items = [item for item in items if item.get('content_type') == 'audio']
            durations = [item.get('metadata', {}).get('duration', 0) for item in audio_items]

            self.statistics['audio'] = {
                'count': len(audio_items),
                'total_duration': sum(durations),
                'avg_duration': sum(durations) / len(durations) if durations else 0,
            }

        if 'text' in content_types:
            text_items = [item for item in items if item.get('content_type') == 'text']
            word_counts = [item.get('metadata', {}).get('word_count', 0) for item in text_items]

            self.statistics['text'] = {
                'count': len(text_items),
                'total_words': sum(word_counts),
                'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            }

        # Store split info
        if 'splits' not in self.metadata:
            self.metadata['splits'] = []

        self.metadata['splits'].append({
            'name': split_name,
            'num_examples': num_examples,
            'size_bytes': total_size,
            'content_types': dict(content_types),
            'mime_types': dict(mime_types),
        })

        logger.info(f"Analyzed {split_name}: {num_examples} examples, {total_size / 1e6:.1f} MB")

    def add_provenance(
        self,
        sources: List[Dict[str, Any]],
        processing_steps: List[str],
        tools: Optional[Dict[str, str]] = None,
    ):
        """
        Add provenance information.

        Args:
            sources: List of source datasets/files
            processing_steps: List of processing steps applied
            tools: Dict of tool name -> version
        """
        self.metadata['provenance'] = {
            'sources': sources,
            'processing_steps': processing_steps,
            'tools': tools or {},
            'created_at': datetime.now().isoformat(),
        }

    def add_curation_step(self, step: str):
        """Add a curation step to the history."""
        self.curation_steps.append({
            'step': step,
            'timestamp': datetime.now().isoformat(),
        })

    def add_quality_metrics(
        self,
        metrics: Dict[str, Any],
    ):
        """Add quality metrics (e.g., dedup stats, toxicity stats)."""
        self.metadata['quality_metrics'] = metrics

    def generate_yaml_header(self) -> str:
        """Generate YAML front matter for dataset card."""
        yaml_data = {
            'dataset_info': {
                'dataset_name': self.dataset_name,
                'features': self._infer_features(),
                'splits': self.metadata.get('splits', []),
                'download_size': sum(split.get('size_bytes', 0) for split in self.metadata.get('splits', [])),
            },
            'license': self.license,
            'tags': self.tags,
            'task_categories': self._infer_task_categories(),
        }

        if self.homepage:
            yaml_data['homepage'] = self.homepage

        return "---\n" + yaml.dump(yaml_data, default_flow_style=False) + "---\n"

    def _infer_features(self) -> List[Dict]:
        """Infer dataset features from statistics."""
        features = []

        if 'images' in self.statistics:
            features.append({'name': 'image', 'dtype': 'image'})

        if 'videos' in self.statistics:
            features.append({'name': 'video', 'dtype': 'video'})

        if 'audio' in self.statistics:
            features.append({'name': 'audio', 'dtype': 'audio'})

        if 'text' in self.statistics:
            features.append({'name': 'text', 'dtype': 'string'})

        # Add common metadata fields
        features.append({'name': 'metadata', 'dtype': 'dict'})

        return features

    def _infer_task_categories(self) -> List[str]:
        """Infer task categories from content types."""
        tasks = []

        if 'images' in self.statistics:
            tasks.extend(['image-classification', 'image-to-text'])

        if 'videos' in self.statistics:
            tasks.extend(['video-classification'])

        if 'audio' in self.statistics:
            tasks.extend(['automatic-speech-recognition'])

        if 'text' in self.statistics:
            tasks.extend(['text-classification', 'text-generation'])

        return tasks

    def generate_markdown_body(self) -> str:
        """Generate markdown body for dataset card."""
        sections = []

        # Header
        sections.append(f"# Dataset Card for {self.dataset_name}\n")

        # Table of contents
        sections.append("## Table of Contents")
        sections.append("- [Dataset Description](#dataset-description)")
        sections.append("- [Dataset Structure](#dataset-structure)")
        sections.append("- [Dataset Creation](#dataset-creation)")
        sections.append("- [Statistics](#statistics)")
        sections.append("- [Provenance](#provenance)")
        if self.license:
            sections.append("- [Licensing](#licensing)")
        if self.citation:
            sections.append("- [Citation](#citation)")
        sections.append("")

        # Description
        sections.append("## Dataset Description\n")
        sections.append(self.description)
        sections.append("")

        # Dataset structure
        sections.append("## Dataset Structure\n")
        sections.append("### Data Instances\n")

        if self.metadata.get('splits'):
            sections.append("The dataset contains the following splits:\n")
            for split in self.metadata['splits']:
                sections.append(f"- **{split['name']}**: {split['num_examples']} examples "
                               f"({split['size_bytes'] / 1e6:.1f} MB)")
            sections.append("")

        sections.append("### Data Fields\n")
        features = self._infer_features()
        for feature in features:
            sections.append(f"- **{feature['name']}**: {feature['dtype']}")
        sections.append("")

        # Statistics
        sections.append("## Statistics\n")

        if 'images' in self.statistics:
            img_stats = self.statistics['images']
            sections.append(f"### Images ({img_stats['count']} items)\n")
            sections.append(f"- Average dimensions: {img_stats['avg_width']:.0f}x{img_stats['avg_height']:.0f}")
            sections.append(f"- Formats: {dict(img_stats['formats'])}")
            sections.append("")

        if 'videos' in self.statistics:
            vid_stats = self.statistics['videos']
            sections.append(f"### Videos ({vid_stats['count']} items)\n")
            sections.append(f"- Total duration: {vid_stats['total_duration'] / 60:.1f} minutes")
            sections.append(f"- Average duration: {vid_stats['avg_duration']:.1f} seconds")
            sections.append(f"- Average FPS: {vid_stats['avg_fps']:.1f}")
            sections.append("")

        if 'audio' in self.statistics:
            aud_stats = self.statistics['audio']
            sections.append(f"### Audio ({aud_stats['count']} items)\n")
            sections.append(f"- Total duration: {aud_stats['total_duration'] / 60:.1f} minutes")
            sections.append(f"- Average duration: {aud_stats['avg_duration']:.1f} seconds")
            sections.append("")

        if 'text' in self.statistics:
            txt_stats = self.statistics['text']
            sections.append(f"### Text ({txt_stats['count']} items)\n")
            sections.append(f"- Total words: {txt_stats['total_words']:,}")
            sections.append(f"- Average words per item: {txt_stats['avg_words']:.0f}")
            sections.append("")

        # Dataset creation
        sections.append("## Dataset Creation\n")

        if self.curation_steps:
            sections.append("### Curation Process\n")
            sections.append("The dataset was created through the following steps:\n")
            for i, step in enumerate(self.curation_steps, 1):
                sections.append(f"{i}. {step['step']}")
            sections.append("")

        # Provenance
        if 'provenance' in self.metadata:
            prov = self.metadata['provenance']
            sections.append("## Provenance\n")

            if prov.get('sources'):
                sections.append("### Source Data\n")
                for source in prov['sources']:
                    sections.append(f"- {source.get('name', 'Unknown')}: {source.get('description', '')}")
                sections.append("")

            if prov.get('processing_steps'):
                sections.append("### Processing Steps\n")
                for step in prov['processing_steps']:
                    sections.append(f"- {step}")
                sections.append("")

            if prov.get('tools'):
                sections.append("### Tools Used\n")
                for tool, version in prov['tools'].items():
                    sections.append(f"- {tool}: {version}")
                sections.append("")

            sections.append(f"**Created**: {prov.get('created_at', 'Unknown')}\n")

        # Quality metrics
        if 'quality_metrics' in self.metadata:
            qm = self.metadata['quality_metrics']
            sections.append("## Quality Metrics\n")

            for metric_name, metric_value in qm.items():
                sections.append(f"- **{metric_name}**: {metric_value}")
            sections.append("")

        # Licensing
        if self.license:
            sections.append("## Licensing\n")
            sections.append(f"This dataset is released under the **{self.license}** license.\n")

        # Citation
        if self.citation:
            sections.append("## Citation\n")
            sections.append("```bibtex")
            sections.append(self.citation)
            sections.append("```\n")

        # Footer
        sections.append("---\n")
        sections.append("*This dataset card was automatically generated by SparkTrainer.*")

        return "\n".join(sections)

    def generate(self, output_path: Optional[str] = None) -> str:
        """
        Generate complete dataset card.

        Args:
            output_path: Optional path to save card to

        Returns:
            Complete dataset card as string
        """
        yaml_header = self.generate_yaml_header()
        markdown_body = self.generate_markdown_body()

        card = yaml_header + "\n" + markdown_body

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(card)
            logger.info(f"Dataset card saved to {output_path}")

        return card


def generate_card_from_manifest(
    manifest_path: str,
    dataset_name: str,
    output_path: str,
    **kwargs
) -> str:
    """
    Convenience function to generate dataset card from manifest.

    Args:
        manifest_path: Path to JSONL manifest
        dataset_name: Name of dataset
        output_path: Path to save card
        **kwargs: Additional arguments for DatasetCardGenerator

    Returns:
        Generated dataset card
    """
    generator = DatasetCardGenerator(
        dataset_name=dataset_name,
        dataset_path=str(Path(manifest_path).parent),
        **kwargs
    )

    generator.analyze_manifest(manifest_path)

    # Add default curation steps
    generator.add_curation_step("Data ingestion from source files")
    generator.add_curation_step("Quality filtering and validation")
    generator.add_curation_step("Manifest generation")

    return generator.generate(output_path)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset card")
    parser.add_argument("manifest", help="Input manifest (JSONL)")
    parser.add_argument("--name", "-n", required=True, help="Dataset name")
    parser.add_argument("--output", "-o", required=True, help="Output path for dataset card")
    parser.add_argument("--description", "-d", help="Dataset description")
    parser.add_argument("--license", "-l", help="License")
    parser.add_argument("--split", default="train", help="Split name")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    card = generate_card_from_manifest(
        manifest_path=args.manifest,
        dataset_name=args.name,
        output_path=args.output,
        description=args.description,
        license=args.license,
    )

    print(f"\nGenerated dataset card ({len(card)} chars)")


if __name__ == "__main__":
    main()
