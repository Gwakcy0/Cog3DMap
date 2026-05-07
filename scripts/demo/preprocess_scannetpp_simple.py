
"""
Simple parallel preprocessing script for ScanNet++ dataset.
Can be run multiple times with different chunk assignments.
Usage:
    CUDA_VISIBLE_DEVICES=0 python preprocess_scannetpp_simple.py --curr_chunk 0 --total_chunks 8
    CUDA_VISIBLE_DEVICES=1 python preprocess_scannetpp_simple.py --curr_chunk 1 --total_chunks 8
    ...
"""

import argparse
from pathlib import Path
from demo_point3r import load_models, preprocess_images
from tqdm import tqdm
import os
from natsort import natsorted

def setup_scannetpp_paths(save_path='./data/media/scannetpp/pointer_memory'):
    """
    Setup paths for ScanNet++ dataset.
    
    Args:
        save_path: Directory where preprocessed pointer data will be saved
        
    Returns:
        input_image_paths: List of paths to image directories for each scene
        pointer_data_paths: List of paths where pointer data will be saved
    """
    base_dir = Path('./data/media/scannetpp')
    
    if not base_dir.exists():
        raise ValueError(f"Dataset directory not found: {base_dir}")
    
    # Get all scene directories
    scene_dirs = natsorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    # Build paths to image directories
    input_image_paths = []
    for scene_dir in scene_dirs:
        image_dir = scene_dir / 'dslr' / 'resized_undistorted_images'
        if image_dir.exists():
            input_image_paths.append(str(image_dir))
        else:
            print(f"Warning: Image directory not found for scene {scene_dir.name}")
    
    # Create save directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build pointer data save paths
    pointer_data_paths = [
        str(save_dir / f"{Path(path).parent.parent.name}.pt")
        for path in input_image_paths
    ]
    
    return input_image_paths, pointer_data_paths

def main():
    parser = argparse.ArgumentParser(description='Preprocess ScanNet++ scenes for a specific GPU')
    parser.add_argument('--lambda-decay', type=float, default=1.0,
                        help='EMA decay factor for embedding merge: updated = lambda * new + (1-lambda) * old (default: 1.0)')
    parser.add_argument('--save-path', type=str, default='./output/scannetpp',
                        help='Output directory path where preprocessed data will be saved (default: ./data/media/scannetpp/pointer_memory)')
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
    args = parser.parse_args()

    lambda_decay = args.lambda_decay
    sample_ct = args.sample_ct
    save_path = args.save_path

    # Get all paths
    input_image_paths, pointer_data_paths = setup_scannetpp_paths(save_path)

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
                preprocess_images(model, processor, min_pixels, max_pixels, point3r_model,
                                  input_images_dir, pointer_data_path, use_viser=False, unload_point3r_model=False,
                                  lambda_decay=lambda_decay, sample_ct=sample_ct, max_memory_tokens=max_mem)
                break
            except Exception as e:
                if max_mem == 10000:
                    print(f"\nFailed to process {input_images_dir}: {e}")
                else:
                    print(f"\nmax_memory_tokens={max_mem} failed for {input_images_dir}, retrying with lower limit: {e}")

    if args.smoke_test:
        print(f"\n{'='*50}")
        print(f"SMOKE TEST COMPLETED!")
        print(f"Processed scene: {Path(local_input_paths[0]).parent.parent.name}")
        print(f"Output: {local_output_paths[0]}")
        print(f"{'='*50}")
    else:
        print(f"\nChunk {args.curr_chunk}: Completed processing {len(local_input_paths)} scenes!")

if __name__ == "__main__":
    main()
