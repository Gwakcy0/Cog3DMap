from pathlib import Path
from demo_point3r import load_models, preprocess_images
from tqdm import tqdm
import torch
import torch.distributed as dist
import os

def setup_scannet_paths():
    base_dir = Path('./data/media/scannet')
    base_dir.mkdir(parents=True, exist_ok=True)

    posed_images_dir = base_dir / 'posed_images'
    input_image_paths = sorted([str(subfolder) for subfolder in posed_images_dir.iterdir() if subfolder.is_dir()])

    pointer_memory_dir = base_dir / 'pointer_memory'
    pointer_memory_dir.mkdir(parents=True, exist_ok=True)

    pointer_data_paths = [
        str(pointer_memory_dir / f"{Path(path).name}.pt")
        for path in input_image_paths
    ]

    return input_image_paths, pointer_data_paths

def init_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0

def main():
    # Initialize distributed environment
    rank, world_size, local_rank = init_distributed()

    # Get all paths - this is CPU operation but read-only, so safe in parallel
    input_image_paths, pointer_data_paths = setup_scannet_paths()

    # Split data across GPUs
    total_scenes = len(input_image_paths)
    scenes_per_gpu = (total_scenes + world_size - 1) // world_size
    start_idx = rank * scenes_per_gpu
    end_idx = min(start_idx + scenes_per_gpu, total_scenes)

    # Get this GPU's subset
    local_input_paths = input_image_paths[start_idx:end_idx]
    local_output_paths = pointer_data_paths[start_idx:end_idx]

    if rank == 0:
        print(f"Total scenes: {total_scenes}")
        print(f"World size (GPUs): {world_size}")
        print(f"Scenes per GPU: ~{scenes_per_gpu}")

    print(f"[GPU {rank}] Processing {len(local_input_paths)} scenes (indices {start_idx} to {end_idx-1})")

    # Synchronize before loading models to avoid race conditions in model cache
    if world_size > 1:
        dist.barrier()

    # Load models with device_map="auto" which handles device placement better
    # The torch.cuda.set_device(local_rank) in init_distributed() ensures
    # that "auto" will use the correct GPU for this process
    model, processor, min_pixels, max_pixels, point3r_model = load_models(device=None)

    # Synchronize after model loading
    if world_size > 1:
        dist.barrier()

    # Process this GPU's subset with progress bar
    for input_images_dir, pointer_data_path in tqdm(
        zip(local_input_paths, local_output_paths),
        desc=f"GPU {rank}",
        total=len(local_input_paths),
        position=rank
    ):
        # Skip if already processed
        if Path(pointer_data_path).exists():
            continue

        preprocess_images(model, processor, min_pixels, max_pixels, point3r_model,
                          input_images_dir, pointer_data_path)

    if rank == 0:
        print(f"\n[GPU {rank}] Completed processing!")

    # Cleanup distributed environment
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
