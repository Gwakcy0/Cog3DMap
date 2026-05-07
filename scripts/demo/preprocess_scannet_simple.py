#!/usr/bin/env python
"""
Simple parallel preprocessing script that can be run multiple times with different GPU assignments.
Usage:
    CUDA_VISIBLE_DEVICES=0 python preprocess_scannet_simple.py --gpu-id 0 --total-gpus 8
    CUDA_VISIBLE_DEVICES=1 python preprocess_scannet_simple.py --gpu-id 1 --total-gpus 8
    ...
"""

import argparse
from pathlib import Path
from demo_point3r import load_models, preprocess_images
from tqdm import tqdm
import os

def setup_scannet_paths(save_path='./data/media/scannet/pointer_memory'):
    base_dir = Path('./data/media/scannet')
    base_dir.mkdir(parents=True, exist_ok=True)

    posed_images_dir = base_dir / 'posed_images'
    input_image_paths = sorted([str(subfolder) for subfolder in posed_images_dir.iterdir() if subfolder.is_dir()])
    # input_image_paths = input_image_paths[::-1]

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    pointer_data_paths = [
        str(save_dir / f"{Path(path).name}.pt")
        for path in input_image_paths
    ]

    return input_image_paths, pointer_data_paths

def main():
    parser = argparse.ArgumentParser(description='Preprocess ScanNet scenes for a specific GPU')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID (0-indexed)')
    parser.add_argument('--total-gpus', type=int, required=True, help='Total number of GPUs')
    parser.add_argument('--lambda-decay', type=float, default=1.0,
                        help='EMA decay factor for embedding merge: updated = lambda * new + (1-lambda) * old (default: 1.0)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Output directory path where preprocessed data will be saved (default: auto-generated from lambda value)')
    parser.add_argument('--sample-ct', type=int, default=32,
                        help='Number of images to uniformly sample per scene (default: 32)')
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help='Output directory name under data/media/scannet/ (default: auto-generated from lambda value)')
    parser.add_argument('--no-merge', action='store_true', default=False,
                        help='Disable spatial merging of memory tokens (use simple concatenation)')
    parser.add_argument('--merge-threshold', type=float, default=None,
                        help='Fixed absolute merge threshold in meters. If set, overrides adaptive threshold. '
                             'Suggested values: 0.05 (fine), 0.15-0.20 (indoor rooms), 0.50 (large scenes). '
                             'Default: None (adaptive, ~5%% of scene diagonal)')
    parser.add_argument('--len-unit', type=int, default=20,
                        help='Number of spatial bins along diagonal for adaptive threshold. '
                             'Only used when --merge-threshold is not set. Default: 20')
    args = parser.parse_args()

    gpu_id = args.gpu_id
    total_gpus = args.total_gpus
    lambda_decay = args.lambda_decay
    sample_ct = args.sample_ct

    # Determine save path
    if args.save_path is not None:
        save_path = args.save_path
    else:
        if lambda_decay == 1.0:
            save_name = 'pointer_memory_qwen3vl'
        else:
            save_name = f'pointer_memory_qwen3vl_lambda{lambda_decay}'
        if "8B" in args.model_path:
            save_name += "_8B"
        save_path = f'./data/media/scannet/{save_name}'

    # Get all paths
    input_image_paths, pointer_data_paths = setup_scannet_paths(save_path)

    # Split data across GPUs
    total_scenes = len(input_image_paths)
    # scenes_per_gpu = (total_scenes + total_gpus - 1) // total_gpus
    # start_idx = gpu_id * scenes_per_gpu
    # end_idx = min(start_idx + scenes_per_gpu, total_scenes)

    # Get this GPU's subset
    # local_input_paths = input_image_paths[start_idx:end_idx]
    # local_output_paths = pointer_data_paths[start_idx:end_idx]
    local_input_paths = input_image_paths[gpu_id::total_gpus]
    local_output_paths = pointer_data_paths[gpu_id::total_gpus]


    print(f"GPU {gpu_id}/{total_gpus-1}: Processing {len(local_input_paths)} scenes")
    print(f"Total scenes in dataset: {total_scenes}")

    # Load models - will use CUDA_VISIBLE_DEVICES=X so it sees only one GPU as cuda:0
    model, processor, min_pixels, max_pixels, point3r_model = load_models(device=None, model_path=args.model_path, use_merge=not args.no_merge, merge_threshold=args.merge_threshold, len_unit=args.len_unit)

    # Process this GPU's subset with progress bar
    for input_images_dir, pointer_data_path in tqdm(
        zip(local_input_paths, local_output_paths),
        desc=f"GPU {gpu_id}",
        total=len(local_input_paths)
    ):
        # Skip if already processed
        if Path(pointer_data_path).exists():
            continue
        for max_mem in [None, 15000, 10000]:
            try:
                preprocess_images(model, processor, min_pixels, max_pixels, point3r_model,
                                  input_images_dir, pointer_data_path, use_viser=False, unload_point3r_model=False,
                                  lambda_decay=lambda_decay, sample_ct=sample_ct, image_extensions=("*.jpg",), max_memory_tokens=max_mem)
                break
            except Exception as e:
                if max_mem == 10000:
                    print(f"\nFailed to process {input_images_dir}: {e}")
                else:
                    print(f"\nmax_memory_tokens={max_mem} failed for {input_images_dir}, retrying with lower limit: {e}")

    print(f"\nGPU {gpu_id}: Completed processing {len(local_input_paths)} scenes!")

if __name__ == "__main__":
    main()
