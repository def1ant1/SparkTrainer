"""
Example: Creating a Multimodal Multistep Dataset

This example demonstrates how to create a complex multimodal dataset that includes:
- Images
- Text (captions, instructions)
- Audio
- Video frames
- Multi-step reasoning chains

The dataset simulates a vision-language-audio reasoning task where the model must:
1. Process visual input (image or video)
2. Process audio input (speech or sound)
3. Understand textual instructions
4. Generate multi-step reasoning
5. Produce a final answer
"""

import os
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Configuration
DATASET_NAME = "multimodal_multistep_vqa"
OUTPUT_DIR = "/home/user/SparkTrainer/datasets"
NUM_SAMPLES = 100


def generate_synthetic_image(idx, width=224, height=224):
    """Generate a synthetic image with random patterns"""
    # Create a random image with geometric patterns
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Create pattern based on index for reproducibility
    random.seed(idx)
    pattern_type = idx % 5

    for i in range(width):
        for j in range(height):
            if pattern_type == 0:  # Gradient
                r = int(255 * (i / width))
                g = int(255 * (j / height))
                b = int(255 * ((i + j) / (width + height)))
            elif pattern_type == 1:  # Checkerboard
                r = 255 if (i // 32 + j // 32) % 2 else 0
                g = 128
                b = 255 if (i // 32 + j // 32) % 2 else 0
            elif pattern_type == 2:  # Circles
                center_x, center_y = width // 2, height // 2
                dist = ((i - center_x)**2 + (j - center_y)**2)**0.5
                r = int(255 * (1 - min(dist / (width/2), 1)))
                g = int(128 * (1 - min(dist / (width/2), 1)))
                b = int(200 * (1 - min(dist / (width/2), 1)))
            elif pattern_type == 3:  # Stripes
                r = 255 if (i + j) % 40 < 20 else 50
                g = 255 if (i - j) % 40 < 20 else 50
                b = 128
            else:  # Random noise
                r = random.randint(100, 255)
                g = random.randint(100, 255)
                b = random.randint(100, 255)

            pixels[i, j] = (r, g, b)

    return img


def generate_reasoning_chain(scenario):
    """Generate multi-step reasoning chain based on scenario"""
    chains = {
        'visual_counting': [
            "Step 1: Identify the objects in the image",
            "Step 2: Count the number of distinct objects",
            "Step 3: Categorize objects by color or shape",
            "Step 4: Calculate the total count"
        ],
        'audio_analysis': [
            "Step 1: Identify the sound source",
            "Step 2: Analyze the frequency and amplitude",
            "Step 3: Determine the context or environment",
            "Step 4: Classify the sound category"
        ],
        'multimodal_fusion': [
            "Step 1: Extract visual features from the image",
            "Step 2: Extract audio features from the sound",
            "Step 3: Align temporal features across modalities",
            "Step 4: Fuse multimodal representations",
            "Step 5: Generate final prediction"
        ],
        'temporal_reasoning': [
            "Step 1: Observe the initial state from frame 1",
            "Step 2: Track changes across frames 2-5",
            "Step 3: Identify the pattern of change",
            "Step 4: Predict the next state",
            "Step 5: Verify prediction consistency"
        ]
    }
    return chains.get(scenario, chains['multimodal_fusion'])


def create_multimodal_sample(idx):
    """Create a single multimodal sample"""
    random.seed(idx)

    # Determine scenario type
    scenarios = ['visual_counting', 'audio_analysis', 'multimodal_fusion', 'temporal_reasoning']
    scenario = scenarios[idx % len(scenarios)]

    # Generate image path (synthetic)
    image_filename = f"sample_{idx:05d}.jpg"

    # Generate text instruction
    instructions = {
        'visual_counting': f"Count the number of distinct objects in the image and describe their arrangement.",
        'audio_analysis': f"Listen to the audio clip and identify the primary sound source and its characteristics.",
        'multimodal_fusion': f"Analyze both the image and audio to determine the context of this scene.",
        'temporal_reasoning': f"Observe the sequence of frames and predict what happens next."
    }
    instruction = instructions[scenario]

    # Generate multi-step reasoning
    reasoning_steps = generate_reasoning_chain(scenario)

    # Generate final answer
    answers = {
        'visual_counting': f"There are {random.randint(3, 10)} objects arranged in a {random.choice(['grid', 'circular', 'linear', 'scattered'])} pattern.",
        'audio_analysis': f"The sound is a {random.choice(['human voice', 'musical instrument', 'mechanical sound', 'natural sound'])} with {random.choice(['high', 'medium', 'low'])} frequency.",
        'multimodal_fusion': f"This appears to be a {random.choice(['indoor', 'outdoor', 'urban', 'natural'])} scene with {random.choice(['peaceful', 'active', 'busy', 'quiet'])} atmosphere.",
        'temporal_reasoning': f"The pattern shows {random.choice(['linear growth', 'cyclic behavior', 'random variation', 'exponential change'])}."
    }
    answer = answers[scenario]

    # Create sample metadata
    sample = {
        'id': f"sample_{idx:05d}",
        'scenario_type': scenario,
        'modalities': ['image', 'text', 'audio'] if scenario in ['multimodal_fusion'] else ['image', 'text'],
        'image_path': image_filename,
        'audio_path': f"audio_{idx:05d}.wav" if scenario in ['audio_analysis', 'multimodal_fusion'] else None,
        'video_frames': [f"frame_{idx:05d}_{i:02d}.jpg" for i in range(5)] if scenario == 'temporal_reasoning' else None,
        'instruction': instruction,
        'reasoning_steps': reasoning_steps,
        'answer': answer,
        'metadata': {
            'difficulty': random.choice(['easy', 'medium', 'hard']),
            'num_steps': len(reasoning_steps),
            'requires_multimodal': scenario in ['multimodal_fusion', 'audio_analysis'],
            'requires_temporal': scenario == 'temporal_reasoning'
        }
    }

    return sample, image_filename


def create_dataset():
    """Create the complete multimodal multistep dataset"""

    # Create output directory
    dataset_dir = Path(OUTPUT_DIR) / DATASET_NAME / "v1"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Create samples
    samples = []
    manifest_path = dataset_dir / "manifest.jsonl"

    print(f"Creating {NUM_SAMPLES} multimodal samples...")

    with open(manifest_path, 'w') as f:
        for idx in range(NUM_SAMPLES):
            # Create sample
            sample, image_filename = create_multimodal_sample(idx)
            samples.append(sample)

            # Generate and save image
            img = generate_synthetic_image(idx)
            img.save(images_dir / image_filename)

            # Write to manifest
            f.write(json.dumps(sample) + '\n')

            if (idx + 1) % 10 == 0:
                print(f"  Created {idx + 1}/{NUM_SAMPLES} samples")

    # Create dataset metadata
    metadata = {
        'name': DATASET_NAME,
        'version': 'v1',
        'description': 'Multimodal multistep reasoning dataset with vision, audio, and text',
        'num_samples': NUM_SAMPLES,
        'modalities': ['image', 'text', 'audio', 'video'],
        'task_types': ['visual_counting', 'audio_analysis', 'multimodal_fusion', 'temporal_reasoning'],
        'features': {
            'multimodal': True,
            'multistep_reasoning': True,
            'temporal': True,
            'avg_reasoning_steps': 4.5
        },
        'statistics': {
            'total_images': NUM_SAMPLES,
            'avg_image_size': '224x224',
            'difficulty_distribution': {
                'easy': NUM_SAMPLES // 3,
                'medium': NUM_SAMPLES // 3,
                'hard': NUM_SAMPLES // 3
            }
        },
        'format': 'jsonl',
        'created_by': 'SparkTrainer Example Generator'
    }

    with open(dataset_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme = """# Multimodal Multistep Reasoning Dataset

## Overview
This dataset contains multimodal samples for training models on complex reasoning tasks.

## Structure
- `manifest.jsonl`: Main dataset file with all samples
- `images/`: Directory containing synthetic images
- `metadata.json`: Dataset metadata and statistics

## Sample Format
Each sample includes:
- **id**: Unique sample identifier
- **scenario_type**: Type of reasoning task
- **modalities**: List of modalities (image, text, audio, video)
- **instruction**: Task instruction
- **reasoning_steps**: Multi-step reasoning chain
- **answer**: Final answer
- **metadata**: Additional sample information

## Task Types
1. **Visual Counting**: Count and describe objects in images
2. **Audio Analysis**: Analyze audio characteristics
3. **Multimodal Fusion**: Combine visual and audio information
4. **Temporal Reasoning**: Reason about sequential frames

## Usage
```python
import json

# Load samples
with open('manifest.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

# Process sample
for sample in samples:
    image_path = sample['image_path']
    instruction = sample['instruction']
    reasoning = sample['reasoning_steps']
    answer = sample['answer']
```

## Statistics
- Total Samples: {num_samples}
- Modalities: Image, Text, Audio, Video
- Average Reasoning Steps: 4.5
- Difficulty Levels: Easy, Medium, Hard
""".format(num_samples=NUM_SAMPLES)

    with open(dataset_dir / 'README.md', 'w') as f:
        f.write(readme)

    print(f"\n✓ Dataset created successfully!")
    print(f"  Location: {dataset_dir}")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Manifest: {manifest_path}")


if __name__ == '__main__':
    create_dataset()
    print("\nDataset creation complete!")
