#!/usr/bin/env python
"""
Simple parallel preprocessing script for robofac video dataset.
Usage:
    CUDA_VISIBLE_DEVICES=0 python preprocess_robofac_simple.py --gpu-id 0 --total-gpus 8
    CUDA_VISIBLE_DEVICES=1 python preprocess_robofac_simple.py --gpu-id 1 --total-gpus 8
    ...
"""

import argparse
from pathlib import Path
from demo_point3r import load_models, preprocess_video
from tqdm import tqdm
import os

def setup_robofac_paths(save_path='./data/media/robofac/pointer_memory_qwen3vl'):
    base_dir = Path('./data/media/robofac')

    # Recursively find all .mp4 files
    video_paths = sorted([str(p) for p in base_dir.rglob('*.mp4')])

    save_dir = Path(save_path)

    # Mirror directory structure: base_dir/rel/path.mp4 -> save_dir/rel/path.pt
    pointer_data_paths = [
        str(save_dir / Path(p).relative_to(base_dir).with_suffix('.pt'))
        for p in video_paths
    ]

    return video_paths, pointer_data_paths

def main():
    parser = argparse.ArgumentParser(description='Preprocess robofac videos for a specific GPU')
    parser.add_argument('--gpu-id', type=int, required=True, help='GPU ID (0-indexed)')
    parser.add_argument('--total-gpus', type=int, required=True, help='Total number of GPUs')
    parser.add_argument('--lambda-decay', type=float, default=1.0,
                        help='EMA decay factor for embedding merge: updated = lambda * new + (1-lambda) * old (default: 1.0)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Output directory path where preprocessed data will be saved (default: auto-generated from lambda value)')
    parser.add_argument('--sample-ct', type=int, default=32,
                        help='Number of frames to uniformly sample per video (default: 32)')
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help='Model path (default: Qwen/Qwen3-VL-4B-Instruct)')
    parser.add_argument('--no-merge', action='store_true', default=False,
                        help='Disable spatial merging of memory tokens (use simple concatenation)')
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
        save_path = f'./data/media/robofac/{save_name}'

    # Get all paths
    video_paths, pointer_data_paths = setup_robofac_paths(save_path)

    # Split data across GPUs (round-robin)
    total_videos = len(video_paths)
    local_video_paths = video_paths[gpu_id::total_gpus]
    local_output_paths = pointer_data_paths[gpu_id::total_gpus]

    print(f"GPU {gpu_id}/{total_gpus-1}: Processing {len(local_video_paths)} videos")
    print(f"Total videos in dataset: {total_videos}")

    # Load models
    model, processor, min_pixels, max_pixels, point3r_model = load_models(device=None, model_path=args.model_path, use_merge=not args.no_merge)

    assert min_pixels == 192 * 32 * 32
    assert max_pixels == 192 * 32 * 32

    # Process this GPU's subset with progress bar
    for video_path, pointer_data_path in tqdm(
        zip(local_video_paths, local_output_paths),
        desc=f"GPU {gpu_id}",
        total=len(local_video_paths)
    ):
        # Skip if already processed
        if Path(pointer_data_path).exists():
            continue
        for max_mem in [None, 15000, 10000]:
            try:
                preprocess_video(model, processor, min_pixels, max_pixels, point3r_model,
                                 video_path, pointer_data_path, use_viser=False, unload_point3r_model=False,
                                 lambda_decay=lambda_decay, sample_ct=sample_ct, max_memory_tokens=max_mem)
                break
            except Exception as e:
                if max_mem == 10000:
                    print(f"\nFailed to process {video_path}: {e}")
                else:
                    print(f"\nmax_memory_tokens={max_mem} failed for {video_path}, retrying with lower limit: {e}")

    print(f"\nGPU {gpu_id}: Completed processing {len(local_video_paths)} videos!")

if __name__ == "__main__":
    main()
