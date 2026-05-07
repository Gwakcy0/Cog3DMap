#!/usr/bin/env python3
"""
Convert scan2cap_train_32frames.json to scan2cap_train_32frames_point3r.json
Replaces multiple <image> tokens with pointer tokens and pointer_data paths.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_scene_id(image_path):
    """
    Extract scene_id from image path.
    Example: 'scannet/posed_images/scene0000_00/01140.jpg' -> 'scene0000_00'
    """
    parts = Path(image_path).parts
    if len(parts) >= 3:
        return parts[2]  # scene0000_00
    return None


def convert_annotation(annotation):
    """
    Convert a single annotation from image-based to pointer-based format.

    Args:
        annotation: Original annotation dict with 'images' and <image> tokens

    Returns:
        Converted annotation dict with 'pointer_data' and pointer tokens
    """
    pre_prompt = "The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n"
    pointer_token = "<|pointer_pad|>"
    new_annotation = annotation.copy()

    # Extract scene_id from first image path
    images = annotation.get('images', [])
    if len(images) == 0:
        print(f"Warning: No images found in annotation {annotation.get('metadata', {})}")
        return None

    scene_id = extract_scene_id(images[0])
    if scene_id is None:
        print(f"Warning: Could not extract scene_id from {images[0]}")
        return None

    # Create pointer_data path
    pointer_data_path = f"scannet/pointer_memory/{scene_id}.pt"
    new_annotation['pointer_data'] = pointer_data_path

    # Remove 'images' field
    if 'images' in new_annotation:
        del new_annotation['images']

    # Update conversations to replace <image> tokens with pointer token
    if 'conversations' in new_annotation:
        new_conversations = []
        for conv in new_annotation['conversations']:
            new_conv = conv.copy()
            value = conv.get('value', '')

            # Replace <image> tokens with a single pointer token
            num_images = value.count('<image>')
            if num_images > 0:
                value_without_images = value.replace('<image>', '')
                new_value = pre_prompt + pointer_token + " " + value_without_images
                new_conv['value'] = new_value

            new_conversations.append(new_conv)

        new_annotation['conversations'] = new_conversations

    return new_annotation


def write_dataset_card(output_dir, splits):
    """Write a HuggingFace dataset card (README.md) with explicit split definitions."""
    data_files = "\n".join(
        f"  - split: {split}\n    path: {filename}" for split, filename in splits
    )
    readme = f"""---
license: apache-2.0
configs:
- config_name: default
  data_files:
{data_files}
---
"""
    with open(Path(output_dir) / "README.md", "w") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(
        description='Convert scan2cap annotations to Point3R format'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/train/scan2cap_train_32frames.json',
        help='Input annotation file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/train/scan2cap_train_32frames_point3r.json',
        help='Output annotation file path'
    )
    args = parser.parse_args()

    # Load input annotations
    print(f"Loading annotations from: {args.input}")
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    with open(input_path, 'r') as f:
        annotations = json.load(f)

    print(f"Total input annotations: {len(annotations)}")

    # Convert annotations
    converted_annotations = []
    skipped = 0

    print("Converting annotations...")
    for annotation in tqdm(annotations):
        converted = convert_annotation(annotation)
        if converted is not None:
            converted_annotations.append(converted)
        else:
            skipped += 1

    print(f"\nConversion complete!")
    print(f"  Converted: {len(converted_annotations)}")
    print(f"  Skipped:   {skipped}")

    # Save output annotations
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(converted_annotations, f, indent=2)

    print("Done!")

    # Print sample conversion
    if len(converted_annotations) > 0:
        print("\n" + "="*80)
        print("SAMPLE CONVERSION (First annotation)")
        print("="*80)

        print("\n--- Original ---")
        orig = annotations[0]
        print(f"Images: {len(orig.get('images', []))} files")
        if 'images' in orig:
            print(f"  First: {orig['images'][0]}")
            print(f"  Last:  {orig['images'][-1]}")
        print(f"Conversation (human): {orig['conversations'][0]['value'][:200]}...")

        print("\n--- Converted ---")
        conv = converted_annotations[0]
        print(f"Pointer data: {conv.get('pointer_data', 'N/A')}")
        print(f"Images field removed: {'images' not in conv}")
        print(f"Conversation (human): {conv['conversations'][0]['value'][:200]}...")
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
