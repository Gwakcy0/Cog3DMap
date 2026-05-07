
"""
Simple parallel preprocessing script for ARKitScenes dataset.
Can be run multiple times with different chunk assignments.
Usage:
    CUDA_VISIBLE_DEVICES=0 python preprocess_arkit_simple.py --curr_chunk 0 --total_chunks 8
    CUDA_VISIBLE_DEVICES=1 python preprocess_arkit_simple.py --curr_chunk 1 --total_chunks 8
    ...
"""

import argparse
import csv
import cv2
from pathlib import Path
from demo_point3r import load_models, preprocess_images
from tqdm import tqdm
import os
import gc
import torch

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_scene_list(metadata_path='./scripts/demo/metadata/arkit_combined.txt'):
    """
    Load scene IDs from metadata file.
    
    Args:
        metadata_path: Path to the metadata file containing scene IDs
        
    Returns:
        List of scene IDs
    """
    with open(metadata_path, 'r') as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    return scene_ids

SKY_DIRECTION_TO_ROT_INDEX = {
    'Up': 0,
    'Left': 1,
    'Down': 2,
    'Right': 3,
}

def load_sky_directions(metadata_csv_path='./data/media/arkitscenes/3dod/metadata.csv'):
    """
    Load sky_direction and fold mapping from arkit_metadata.csv.

    Returns:
        sky_directions: dict of video_id (str) -> sky_direction (str)
        folds: dict of video_id (str) -> fold (str, 'Training' or 'Validation')
    """
    sky_directions = {}
    folds = {}
    with open(metadata_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sky_directions[row['video_id']] = row['sky_direction']
            folds[row['video_id']] = row['fold']
    return sky_directions, folds

def rotate_image(im, rot_index):
    if rot_index == 0:
        return im
    elif rot_index == 1:
        return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    elif rot_index == 2:
        return cv2.rotate(im, cv2.ROTATE_180)
    elif rot_index == 3:
        return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return im

def rotate_scene_images(metadata_path='./scripts/demo/metadata/arkit_combined.txt',
                        metadata_csv_path='./data/media/arkitscenes/3dod/metadata.csv',
                        sample_ct=32):
    """
    For each scene in the metadata list, rotate images from lowres_wide/
    into lowres_wide_rotated/ based on the sky_direction in arkit_metadata.csv.
    Skips scenes that already have a lowres_wide_rotated/ directory.
    """
    base_dirs = {
        'Training':   Path('./data/media/arkitscenes/3dod/Training'),
        'Validation': Path('./data/media/arkitscenes/3dod/Validation'),
    }
    scene_ids = load_scene_list(metadata_path)
    sky_directions, folds = load_sky_directions(metadata_csv_path)

    for scene_id in tqdm(scene_ids, desc="Rotating images"):
        fold = folds.get(scene_id, 'Training')
        base_dir = base_dirs[fold]
        src_dir = base_dir / scene_id / f'{scene_id}_frames' / 'lowres_wide'
        dst_dir = base_dir / scene_id / f'{scene_id}_frames' / 'lowres_wide_rotated'

        if dst_dir.exists():
            continue
        if not src_dir.exists():
            print(f"Warning: lowres_wide not found for scene {scene_id}, skipping rotation")
            continue

        sky_dir = sky_directions.get(scene_id, 'Up')
        rot_index = SKY_DIRECTION_TO_ROT_INDEX.get(sky_dir, 0)

        dst_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(src_dir.glob('*.png'))
        if len(image_paths) > sample_ct:
            step = len(image_paths) / sample_ct
            frames_indices = [int(i * step) for i in range(sample_ct)]
            image_paths = [image_paths[idx] for idx in frames_indices]
        for img_path in image_paths:
            im = cv2.imread(str(img_path))
            rotated = rotate_image(im, rot_index)
            cv2.imwrite(str(dst_dir / img_path.name), rotated)

def setup_arkit_paths(save_path='./output/arkit', metadata_path='./scripts/demo/metadata/arkit_combined.txt'):
    """
    Setup paths for ARKitScenes dataset.

    Args:
        save_path: Directory where preprocessed pointer data will be saved
        metadata_path: Path to the metadata file containing scene IDs

    Returns:
        input_image_paths: List of paths to image directories for each scene
        pointer_data_paths: List of paths where pointer data will be saved
    """
    train_base_dir = Path('./data/media/arkitscenes/3dod/Training')
    val_base_dir   = Path('./data/media/arkitscenes/3dod/Validation')

    # At least one split should exist
    if not train_base_dir.exists() and not val_base_dir.exists():
        raise ValueError(
            f"Dataset directory not found. Checked:\n  - {train_base_dir}\n  - {val_base_dir}"
        )

    # Load scene list from metadata
    scene_ids = load_scene_list(metadata_path)

    # Build paths to image directories (try Training first, then Validation)
    input_image_paths = []
    valid_scene_ids = []

    for scene_id in scene_ids:
        rel = Path(scene_id) / f'{scene_id}_frames' / 'lowres_wide_rotated'

        train_dir = train_base_dir / rel
        val_dir   = val_base_dir / rel

        if train_base_dir.exists() and train_dir.exists():
            input_image_paths.append(str(train_dir))
            valid_scene_ids.append(scene_id)
        elif val_base_dir.exists() and val_dir.exists():
            input_image_paths.append(str(val_dir))
            valid_scene_ids.append(scene_id)
        else:
            print(f"Warning: Image directory not found for scene {scene_id} in Training/Validation")

    # Create save directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build pointer data save paths
    pointer_data_paths = [str(save_dir / f"{scene_id}.pt") for scene_id in valid_scene_ids]

    return input_image_paths, pointer_data_paths

def main():
    parser = argparse.ArgumentParser(description='Preprocess ARKitScenes scenes for a specific GPU')
    parser.add_argument('--lambda-decay', type=float, default=1.0,
                        help='EMA decay factor for embedding merge: updated = lambda * new + (1-lambda) * old (default: 1.0)')
    parser.add_argument('--save-path', type=str, default='./output/arkit',
                        help='Output directory path where preprocessed data will be saved (default: ./output/arkit)')
    parser.add_argument('--metadata-path', type=str, default='./scripts/demo/metadata/arkit_combined.txt',
                        help='Path to the metadata file containing scene IDs (default: ./scripts/demo/metadata/arkit_combined.txt)')
    parser.add_argument('--sample-ct', type=int, default=32,
                        help='Number of images to uniformly sample per scene (default: 32)')
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help='Path to the model (default: Qwen/Qwen3-VL-4B-Instruct)')
    parser.add_argument("--curr_chunk", type=int, default=0,
                        help='Current chunk index (0-indexed)')
    parser.add_argument("--total_chunks", type=int, default=1,
                        help='Total number of chunks')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run smoke test on only the first scene')
    parser.add_argument('--no-merge', action='store_true', default=False,
                        help='Disable spatial merging of memory tokens (use simple concatenation)')
    parser.add_argument('--metadata-csv', type=str, default='./data/media/arkitscenes/3dod/metadata.csv',
                        help='Path to arkit_metadata.csv with sky_direction info (default: ./data/media/arkitscenes/3dod/metadata.csv)')
    args = parser.parse_args()

    lambda_decay = args.lambda_decay
    sample_ct = args.sample_ct
    save_path = args.save_path

    # Rotate images based on sky_direction (skips already-rotated scenes)
    rotate_scene_images(args.metadata_path, args.metadata_csv, sample_ct=sample_ct)

    # Get all paths
    input_image_paths, pointer_data_paths = setup_arkit_paths(save_path, args.metadata_path)

    # Smoke test: only process first scene
    if args.smoke_test:
        print("=" * 50)
        print("SMOKE TEST MODE: Processing only the first scene")
        print("=" * 50)
        local_input_paths = [input_image_paths[0]]
        local_output_paths = [pointer_data_paths[0]]
    else:
        # Split data across GPUs
        total_scenes = len(input_image_paths)
        local_input_paths = input_image_paths[args.curr_chunk::args.total_chunks]
        local_output_paths = pointer_data_paths[args.curr_chunk::args.total_chunks]
        print(f"Processing {len(local_input_paths)} scenes")
        print(f"Total scenes in dataset: {total_scenes}")

    # Load models - will use CUDA_VISIBLE_DEVICES=X so it sees only one GPU as cuda:0
    model, processor, min_pixels, max_pixels, point3r_model = load_models(device=None, model_path=args.model_path, use_merge=not args.no_merge)

    # Process this GPU's subset with progress bar
    desc = "Smoke Test" if args.smoke_test else f"Chunk {args.curr_chunk}/{args.total_chunks}"
    for input_images_dir, pointer_data_path in tqdm(
        zip(local_input_paths, local_output_paths),
        desc=desc,
        total=len(local_input_paths)
    ):
        # Skip if already processed
        if Path(pointer_data_path).exists():
            print(f"\nSkipping (already exists): {pointer_data_path}")
            continue
        for max_mem in [None, 15000, 10000]:
            try:
                with torch.inference_mode():
                    preprocess_images(model, processor, min_pixels, max_pixels, point3r_model,
                                    input_images_dir, pointer_data_path, use_viser=False, unload_point3r_model=False,
                                    lambda_decay=lambda_decay, sample_ct=sample_ct, max_memory_tokens=max_mem)
                break
            except Exception as e:
                if max_mem == 10000:
                    print(f"\nFailed to process {input_images_dir}: {e}")
                else:
                    print(f"\nmax_memory_tokens={max_mem} failed for {input_images_dir}, retrying with lower limit: {e}")
                cleanup_cuda()

    if args.smoke_test:
        print(f"\n{'='*50}")
        print(f"SMOKE TEST COMPLETED!")
        print(f"Processed scene: {Path(local_input_paths[0]).parent.name}")
        print(f"Output: {local_output_paths[0]}")
        print(f"{'='*50}")
    else:
        print(f"\nChunk {args.curr_chunk}: Completed processing {len(local_input_paths)} scenes!")

if __name__ == "__main__":
    main()
