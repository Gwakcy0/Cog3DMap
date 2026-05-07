"""
Utility functions for working with camera poses from Point3R.

This module provides helper functions to convert and manipulate camera poses
extracted by extract_pointer_memory() when pose_head=True.
"""

import torch
import numpy as np
from .utils.camera import pose_encoding_to_camera, quaternion_to_matrix


def poses_to_c2w_matrices(camera_poses):
    """
    Convert 7D pose encodings to 4x4 camera-to-world transformation matrices.

    Args:
        camera_poses: Tensor of shape (num_frames, 7) where each row is [tx, ty, tz, qw, qx, qy, qz]
                     Can also be a single pose of shape (7,)

    Returns:
        c2w_matrices: Tensor of shape (num_frames, 4, 4) containing camera-to-world transformation matrices
                     Each matrix transforms points from camera coordinates to world coordinates
                     Convention: OpenCV (X-right, Y-down, Z-forward)

    Example:
        >>> camera_poses = pointer_data['camera_poses']  # Shape: (10, 7)
        >>> c2w = poses_to_c2w_matrices(camera_poses)
        >>> # Transform a 3D point from camera 0 to world coordinates
        >>> point_cam = torch.tensor([0.0, 0.0, 1.0, 1.0])  # Homogeneous coordinates
        >>> point_world = c2w[0] @ point_cam
    """
    if camera_poses.dim() == 1:
        camera_poses = camera_poses.unsqueeze(0)

    return pose_encoding_to_camera(camera_poses, pose_encoding_type='absT_quaR')


def extract_translation_rotation(camera_poses):
    """
    Extract translation vectors and rotation matrices from 7D pose encodings.

    Args:
        camera_poses: Tensor of shape (num_frames, 7) or (7,)

    Returns:
        translations: Tensor of shape (num_frames, 3) - translation vectors [tx, ty, tz]
        rotations: Tensor of shape (num_frames, 3, 3) - rotation matrices

    Example:
        >>> camera_poses = pointer_data['camera_poses']
        >>> translations, rotations = extract_translation_rotation(camera_poses)
        >>> print(f"Camera 0 position: {translations[0]}")
        >>> print(f"Camera 0 orientation:\n{rotations[0]}")
    """
    if camera_poses.dim() == 1:
        camera_poses = camera_poses.unsqueeze(0)

    translations = camera_poses[:, :3]  # Shape: (num_frames, 3)
    quaternions = camera_poses[:, 3:7]  # Shape: (num_frames, 4)
    rotations = quaternion_to_matrix(quaternions)  # Shape: (num_frames, 3, 3)

    return translations, rotations


def compute_relative_pose(pose1, pose2):
    """
    Compute the relative pose from camera 1 to camera 2.

    Args:
        pose1: Tensor of shape (7,) - first camera pose [tx, ty, tz, qw, qx, qy, qz]
        pose2: Tensor of shape (7,) - second camera pose

    Returns:
        relative_pose: Tensor of shape (7,) - relative pose from camera 1 to camera 2

    Example:
        >>> camera_poses = pointer_data['camera_poses']
        >>> rel_pose = compute_relative_pose(camera_poses[0], camera_poses[1])
        >>> print(f"Relative translation: {rel_pose[:3]}")
    """
    from .utils.camera import relative_pose_absT_quatR

    t1, q1 = pose1[:3], pose1[3:7]
    t2, q2 = pose2[:3], pose2[3:7]

    t_rel, q_rel = relative_pose_absT_quatR(
        t1.unsqueeze(0), q1.unsqueeze(0),
        t2.unsqueeze(0), q2.unsqueeze(0)
    )

    return torch.cat([t_rel[0], q_rel[0]], dim=0)


def visualize_camera_trajectory(camera_poses, save_path=None):
    """
    Visualize camera trajectory in 3D.

    Args:
        camera_poses: Tensor of shape (num_frames, 7)
        save_path: Optional path to save the visualization

    Returns:
        fig: matplotlib figure object

    Example:
        >>> camera_poses = pointer_data['camera_poses']
        >>> fig = visualize_camera_trajectory(camera_poses, 'trajectory.png')
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    if isinstance(camera_poses, torch.Tensor):
        camera_poses = camera_poses.cpu().numpy()

    # Extract positions
    positions = camera_poses[:, :3]  # (num_frames, 3)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, label='Camera trajectory')

    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c='r', marker='o', s=50, label='Camera positions')

    # Mark start and end
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
              c='g', marker='*', s=200, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
              c='orange', marker='*', s=200, label='End')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Trajectory ({len(positions)} frames)')
    ax.legend()

    # Equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {save_path}")

    return fig


def get_camera_info_summary(camera_poses):
    """
    Get a summary of camera pose statistics.

    Args:
        camera_poses: Tensor of shape (num_frames, 7)

    Returns:
        dict: Dictionary containing statistics about the camera trajectory

    Example:
        >>> camera_poses = pointer_data['camera_poses']
        >>> info = get_camera_info_summary(camera_poses)
        >>> print(f"Average camera distance: {info['avg_distance_from_origin']:.3f}")
    """
    if isinstance(camera_poses, torch.Tensor):
        camera_poses = camera_poses.cpu()

    translations = camera_poses[:, :3]

    # Compute statistics
    distances = torch.norm(translations, dim=1)

    info = {
        'num_frames': len(camera_poses),
        'translation_mean': translations.mean(dim=0).tolist(),
        'translation_std': translations.std(dim=0).tolist(),
        'translation_range': {
            'x': [translations[:, 0].min().item(), translations[:, 0].max().item()],
            'y': [translations[:, 1].min().item(), translations[:, 1].max().item()],
            'z': [translations[:, 2].min().item(), translations[:, 2].max().item()],
        },
        'avg_distance_from_origin': distances.mean().item(),
        'max_distance_from_origin': distances.max().item(),
        'min_distance_from_origin': distances.min().item(),
    }

    # Compute trajectory length
    if len(camera_poses) > 1:
        deltas = translations[1:] - translations[:-1]
        segment_lengths = torch.norm(deltas, dim=1)
        info['trajectory_length'] = segment_lengths.sum().item()
        info['avg_step_length'] = segment_lengths.mean().item()

    return info


if __name__ == "__main__":
    print("Camera Pose Utilities for Point3R")
    print("=" * 70)
    print("\nExample usage:")
    print("""
    from extract_pointer_memory import extract_pointer_memory
    from camera_pose_utils import (
        poses_to_c2w_matrices,
        extract_translation_rotation,
        visualize_camera_trajectory,
        get_camera_info_summary
    )

    # Extract pointer memory and camera poses
    pointer_data = extract_pointer_memory(image_inputs, point3r_model)

    if 'camera_poses' in pointer_data:
        camera_poses = pointer_data['camera_poses']

        # Convert to 4x4 matrices
        c2w_matrices = poses_to_c2w_matrices(camera_poses)

        # Extract translation and rotation
        translations, rotations = extract_translation_rotation(camera_poses)

        # Get statistics
        info = get_camera_info_summary(camera_poses)
        print(f"Number of frames: {info['num_frames']}")
        print(f"Trajectory length: {info['trajectory_length']:.2f}")

        # Visualize
        visualize_camera_trajectory(camera_poses, 'camera_trajectory.png')
    """)
