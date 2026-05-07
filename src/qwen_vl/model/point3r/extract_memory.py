"""
Function to extract pointer memory from image inputs using Point3R model.

This module provides utilities to convert image inputs (from qwen_vl_utils)
into Point3R memory features that can be used with the Point3R-enhanced model.
"""

import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as tvf
import sys

from .inference import inference, get_pred_pts3d
from .point3r import LocalMemory
from .utils.geometry import geotrf
from typing import List, Dict


def prepare_images_for_point3r(image_inputs, target_size=(640, 480), crop_border=20):
    """
    Prepare images for Point3R processing.

    This function processes images similar to the ScanNetDataset pattern:
    - Crops borders if specified
    - Resizes to target dimensions
    - Converts to normalized tensors

    Args:
        image_inputs: List of images (PIL Images, numpy arrays, or file paths)
        target_size: Tuple of (width, height) for resizing (default: (640, 480))
        crop_border: Number of pixels to crop from each edge (default: 20)

    Returns:
        list: List of view dictionaries containing:
            - 'img': Normalized image tensor (3, H, W)
            - 'true_shape': Tensor of shape (2,) with [height, width]
            - 'img_mask': Boolean tensor indicating valid image
    """
    views = []

    for img_input in image_inputs:
        # Convert to PIL Image if needed
        if isinstance(img_input, str):
            image = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            image = Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            image = img_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_input)}")

        # Crop borders if specified
        if crop_border > 0:
            image = ImageOps.crop(image, border=crop_border)

        # Resize to target dimensions
        image = image.resize(target_size, Image.LANCZOS)

        # Convert to tensor [0, 1] range (matching reference implementation)
        img_tensor = tvf.ToTensor()(image)  # Shape: (3, H, W), range [0, 1]

        # Add batch dimension to match Point3R's expectation
        # Point3R expects: (batch_size, 3, H, W)
        img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

        # Create true_shape tensor [height, width] with batch dimension
        true_shape = torch.tensor([[image.height, image.width]], dtype=torch.int32)  # Shape: (1, 2)

        # Create img_mask with batch dimension
        img_mask = torch.tensor([True], dtype=torch.bool)  # Shape: (1,)

        # Create view dictionary
        view = {
            "img": img_tensor,
            "true_shape": true_shape,
            "img_mask": img_mask,
        }
        views.append(view)

    return views


def visualize_point3r_viser(
    pointer_data,
    annotation_result=None,
    scannet_pth_path=None,
    scannet_pose_paths=None,
):
    """
    Launch an interactive viser 3D visualization of Point3R outputs.

    Call this after extract_pointer_memory() with its return dict.

    Args:
        pointer_data: Return dict from extract_pointer_memory (must include '_point3r_outputs').
        annotation_result: Output from extract_box_and_coordinates_from_scan2cap for visualization.
        scannet_pth_path: Path to ScanNet .pth file for GT point cloud visualization.
        scannet_pose_paths: List of paths to ScanNet pose .txt files with camera poses.
    """
    import viser
    import viser.transforms as tf
    import matplotlib.cm as cm
    import time

    outputs = pointer_data['_point3r_outputs']
    pointer_positions = pointer_data['pointer_positions']
    camera_poses = pointer_data.get('camera_poses')

    viser_start_time = time.time()
    server = viser.ViserServer()

    if annotation_result is not None:
        # Get transformation matrices from the first element (already computed as numpy arrays)
        first_elem = annotation_result['all_elements'][0]
        global2cam = first_elem['global2cam']
        ref_cam2global = first_elem['ref_cam2global']
        axis_align_matrix = first_elem['axis_align']

        # Visualize ScanNet pose files if provided
        if scannet_pose_paths is not None and len(scannet_pose_paths) > 0:
            print(f"Loading {len(scannet_pose_paths)} ScanNet pose files...")
            gt_frustums: List[viser.CameraFrustumHandle] = []
            for i, pose_path in enumerate(scannet_pose_paths):
                try:
                    # Load 4x4 pose matrix from txt file
                    pose_matrix = np.loadtxt(pose_path)
                    if pose_matrix.shape == (4, 4):
                        aligned_pose_matrix = axis_align_matrix @ pose_matrix
                        aligned_pose_se3 = tf.SE3.from_matrix(aligned_pose_matrix)
                        h, w = 480, 640
                        fy = 1.1 * h
                        fov = 2 * np.arctan2(h / 2, fy)
                        gt_frustum = server.scene.add_camera_frustum(
                            f"gt_camera_{i}",
                            fov=fov,
                            aspect=w / h,
                            scale=0.05,
                            image=None,
                            line_width=1.0,
                            color=(255, 165, 0),  # Orange color for GT poses
                            position=aligned_pose_se3.translation(),
                            wxyz=aligned_pose_se3.rotation().wxyz
                        )
                        gt_frustums.append(gt_frustum)
                    if i == 0:
                        ref_cam2global = pose_matrix
                        ref_cam2global_se3 = tf.SE3.from_matrix(ref_cam2global)
                        print('Overwrite ref_cam2global')
                        print(f"GT ref.frame camera pose: {pose_se3}")
                        print(f"Given ref.frame camera pose: {ref_cam2global_se3}")


                except Exception as e:
                    print(f"  Warning: Failed to load pose from {pose_path}: {e}")
            print(f"  Added {len(gt_frustums)} GT camera frustums (orange)")


        extrinsic = axis_align_matrix @ ref_cam2global
        extrinsic_se3 = tf.SE3.from_matrix(extrinsic)
        align_wxyz = extrinsic_se3.rotation().wxyz
        align_pos = extrinsic_se3.translation()
        print('Current extrinsic:', align_wxyz, align_pos, sep='\n')
    else:
        axis_align_matrix = np.eye(4)
        extrinsic_se3 = tf.SE3.from_matrix(np.eye(4))
        align_wxyz = extrinsic_se3.rotation().wxyz
        align_pos = extrinsic_se3.translation()
        print('No annotation given.')

    # Store per-frame data for timestamp visualization
    per_frame_data = []
    num_frames = len(outputs['pred'])

    for idx, (pred, view) in enumerate(zip(outputs['pred'], outputs['views'])):
        pts_3d = get_pred_pts3d(None, pred, use_pose=True)

        # Original RGB image for frustum display (H, W, 3)
        rgb_image = view['img'].permute(0, 2, 3, 1).squeeze(0)
        rgb_image_np = (rgb_image.detach().cpu().numpy() * 255).astype(np.uint8)

        # Points and colors (NO quantile filtering for interpretability)
        pts_3d_np = pts_3d.detach().cpu().numpy().reshape(-1, 3)
        color_rgb = rgb_image_np.reshape(-1, 3)

        # Viridis color for this frame's points
        norm_idx = idx / max(num_frames - 1, 1)
        viridis_rgba = cm.viridis(norm_idx)
        color_timestamp = np.full_like(color_rgb, (np.array(viridis_rgba[:3]) * 255).astype(np.uint8))

        per_frame_data.append({
            'pts_3d': pts_3d_np,
            'colors_rgb': color_rgb,
            'colors_timestamp': color_timestamp,
            'rgb_image': rgb_image_np,
            'frame_idx': idx,
        })

    # === GUI Controls ===
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames-1, step=1, initial_value=num_frames-1)
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider("FPS", min=0.5, max=100, step=0.5, initial_value=1)
        gui_accumulative = server.gui.add_checkbox("Accumulative Mode", True)
        gui_stride = server.gui.add_slider("Stride", min=1, max=max(num_frames, 1), step=1, initial_value=1)
        gui_num_frames_visible = server.gui.add_slider("Num Frames Visible", min=1, max=max(num_frames, 1), step=1, initial_value=max(num_frames, 1))

    with server.gui.add_folder("Visualization"):
        gui_color_mode = server.gui.add_dropdown("Color Mode", options=["Original RGB", "Timestamp (viridis)"], initial_value="Original RGB")
        gui_show_frustums = server.gui.add_checkbox("Show Frustums", True)
        gui_point_size = server.gui.add_slider("Point Size", min=0.0001, max=0.03, step=0.0005, initial_value=0.005)
        gui_frustum_scale = server.gui.add_slider("Frustum Scale", min=0.01, max=0.2, step=0.01, initial_value=0.05)

    # Create parent frame for all timesteps
    server.scene.add_frame("/frames", show_axes=False, wxyz=align_wxyz, position=align_pos)

    frame_nodes: List[viser.FrameHandle] = []
    point_cloud_handles: Dict[int, Dict[str, any]] = {}
    frustum_handles: List[viser.CameraFrustumHandle] = []

    for frame_data in per_frame_data:
        idx = frame_data['frame_idx']

        # Frame node for this timestep
        frame_node = server.scene.add_frame(f"/frames/t{idx}", show_axes=False)
        frame_nodes.append(frame_node)

        # Point cloud with RGB colors
        pc_rgb = server.scene.add_point_cloud(
            name=f"/frames/t{idx}/points_rgb",
            points=frame_data['pts_3d'],
            colors=frame_data['colors_rgb'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=True,
        )

        # Point cloud with timestamp colors (initially hidden)
        pc_timestamp = server.scene.add_point_cloud(
            name=f"/frames/t{idx}/points_timestamp",
            points=frame_data['pts_3d'],
            colors=frame_data['colors_timestamp'],
            point_size=gui_point_size.value,
            point_shape="rounded",
            visible=False,
        )

        point_cloud_handles[idx] = {'rgb': pc_rgb, 'timestamp': pc_timestamp}

        # Camera frustum with RGB image and viridis-colored edge
        if camera_poses is not None:
            pose = camera_poses[idx].numpy()
            pose_se3 = extrinsic_se3 @ tf.SE3(np.concatenate([pose[3:], pose[:3]]))

            norm_idx = idx / max(num_frames - 1, 1)
            frustum_color = tuple((np.array(cm.viridis(norm_idx)[:3]) * 255).astype(int))

            h, w = frame_data['rgb_image'].shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = server.scene.add_camera_frustum(
                f"/frames/t{idx}/frustum",
                fov=fov,
                aspect=w / h,
                scale=gui_frustum_scale.value,
                image=frame_data['rgb_image'],
                line_width=2.0,
                color=frustum_color,
                position=pose_se3.translation(),
                wxyz=pose_se3.rotation().wxyz,
            )
            frustum_handles.append(frustum)


    server.scene.add_point_cloud(
        name=f"pointer_memory_anchor",
        points=pointer_positions.numpy(),
        colors=(255, 0, 0),
        point_size=0.02,
        visible=True,
        wxyz=align_wxyz,
        position=align_pos

    )

    # Visualize annotation data if provided
    point_array = np.array([[0, 0, 0]])
    if annotation_result is not None:
        # Get all elements from annotation result
        all_elements = annotation_result.get('all_elements', [])
        framewise_elements = annotation_result.get('by_pointer_data')
        for elem in all_elements:
            obj_id = elem.get('metadata', {}).get('object_id', 'unknown')
            box_center = elem.get('box_center')
            transformed_box_center = elem.get('transformed_center')
            if transformed_box_center is not None:
                pos = tuple(transformed_box_center)
                server.scene.add_point_cloud(
                    name=f'obj_{obj_id} (camera_aligned)',
                    points=point_array,
                    point_size=0.05,
                    colors=(0, 255, 0),
                    position=pos,
                    visible=False
                )

    # Visualize ScanNet GT point cloud if path provided
    if scannet_pth_path is not None:
        import os
        if os.path.exists(scannet_pth_path):
            print(f"Loading ScanNet GT point cloud from {scannet_pth_path}...")
            gt_data = torch.load(scannet_pth_path, weights_only=False)

            gt_xyz = gt_data['xyz']
            gt_rgb = gt_data['rgb']

            server.scene.add_point_cloud(
                name="gt_point_cloud",
                points=gt_xyz,
                colors=gt_rgb,
                point_size=0.01,
                visible=True
            )
            print(f"  Added GT point cloud: {gt_xyz.shape[0]} points")

            # Add AABBs if available
            if 'aabb_corner_xyz' in gt_data and 'aabb_obj_ids' in gt_data:
                aabb_corners = gt_data['aabb_corner_xyz']
                aabb_obj_ids = gt_data['aabb_obj_ids']

                aabb_colors = [
                    (255, 100, 100),  # Light red
                    (100, 255, 100),  # Light green
                    (100, 100, 255),  # Light blue
                    (255, 255, 100),  # Yellow
                    (255, 100, 255),  # Magenta
                    (100, 255, 255),  # Cyan
                ]

                # AABB edges: 12 edges connecting 8 corners
                edges = [
                    (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
                    (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
                ]

                for i, (obj_id, corners) in enumerate(zip(aabb_obj_ids, aabb_corners)):
                    color = aabb_colors[i % len(aabb_colors)]
                    edge_points = []
                    for e0, e1 in edges:
                        edge_points.append(np.array([corners[e0], corners[e1]]))
                    edge_points = np.stack(edge_points, axis=0)

                    server.scene.add_line_segments(
                        name=f"gt_aabb_{obj_id}",
                        points=edge_points,
                        colors=color,
                        line_width=2.0,
                        visible=True
                    )
                print(f"  Added {len(aabb_obj_ids)} GT AABBs")
        else:
            print(f"Warning: ScanNet GT path not found: {scannet_pth_path}")

    # === Event Handlers ===
    def update_frame_visibility():
        """Update frame visibility based on current mode and timestep."""
        current = gui_timestep.value
        stride = gui_stride.value
        max_visible = gui_num_frames_visible.value
        with server.atomic():
            for i, frame_node in enumerate(frame_nodes):
                if gui_accumulative.value:
                    in_range = (i <= current) and (i % stride == 0)
                    if gui_playing.value:
                        in_range = in_range and (i > current - max_visible * stride)
                    frame_node.visible = in_range
                else:
                    frame_node.visible = (i == current)
        server.flush()

    @gui_next_frame.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_timestep.on_update
    def _(_):
        update_frame_visibility()

    @gui_accumulative.on_update
    def _(_):
        update_frame_visibility()

    @gui_stride.on_update
    def _(_):
        update_frame_visibility()

    @gui_color_mode.on_update
    def _(_):
        use_timestamp = (gui_color_mode.value == "Timestamp (viridis)")
        with server.atomic():
            for idx in point_cloud_handles:
                point_cloud_handles[idx]['rgb'].visible = not use_timestamp
                point_cloud_handles[idx]['timestamp'].visible = use_timestamp
        server.flush()

    @gui_num_frames_visible.on_update
    def _(_):
        update_frame_visibility()

    @gui_show_frustums.on_update
    def _(_):
        with server.atomic():
            for frustum in frustum_handles:
                frustum.visible = gui_show_frustums.value

    @gui_point_size.on_update
    def _(_):
        use_timestamp = (gui_color_mode.value == "Timestamp (viridis)")
        with server.atomic():
            for idx in point_cloud_handles:
                fd = per_frame_data[idx]
                point_cloud_handles[idx]['rgb'] = server.scene.add_point_cloud(
                    name=f"/frames/t{idx}/points_rgb",
                    points=fd['pts_3d'],
                    colors=fd['colors_rgb'],
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                    visible=not use_timestamp,
                )
                point_cloud_handles[idx]['timestamp'] = server.scene.add_point_cloud(
                    name=f"/frames/t{idx}/points_timestamp",
                    points=fd['pts_3d'],
                    colors=fd['colors_timestamp'],
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                    visible=use_timestamp,
                )
        server.flush()

    # Initialize visibility
    update_frame_visibility()

    print(f"\nViser server running at: http://localhost:8080")
    print("Press Ctrl+C to stop visualization and continue...")

    try:
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames
            time.sleep(1.0 / gui_framerate.value)
    except KeyboardInterrupt:
        viser_elapsed = time.time() - viser_start_time
        print(f"\nVisualization stopped after {viser_elapsed:.2f} seconds")
        print(f"(Subtract this from total preprocessing time for actual processing time)")


def extract_pointer_memory(
    image_inputs,
    point3r_model,
    image_embeds=None,
    grid_thw=None,
    deepstack_image_embeds=None,
    device='cuda',
    no_crop=False,
    full_seq=False,
    size=512,
    verbose=True,
    lambda_decay=1.0,
    max_memory_tokens=None,
    frames_indices=None,
):
    """
    Extract pointer memory from image inputs using Point3R model.

    This function processes images through Point3R to generate memory features
    that can be used with Qwen2_5_VLForConditionalGenerationWithPoint3R.

    Args:
        image_inputs: List of image inputs (can be PIL Images, file paths, or numpy arrays)
        point3r_model: Initialized Point3R model
        device: Device to run inference on (default: 'cuda')
        no_crop: If True, resize instead of crop (default: False)
        full_seq: If True, process full sequence mode (default: False)
        size: Target image size (default: 512)
        verbose: Print progress information (default: True)
        annotation_result: Output from extract_box_and_coordinates_from_scan2cap containing
                          gt_box, transformed_center, and transformation matrices for viser
                          visualization (default: None)
        scannet_pth_path: Path to ScanNet .pth file containing ground truth point cloud
                         (xyz, rgb, aabb_corner_xyz, aabb_obj_ids) for visualization (default: None)
        scannet_pose_paths: List of paths to ScanNet pose .txt files containing camera poses
                           (default: None)

    Returns:
        dict: Dictionary containing:
            - 'pointer_memory_embeds': Tensor of shape (num_pointers, 2048) or (num_pointers, 2560)
                                      Qwen image embeddings aligned with memory (used for LLM input)
            - 'pointer_positions': Tensor of shape (num_pointers, 3)
                                  3D positions (x, y, z) for each pointer in world coordinates
            - 'memory_feat': (Optional) Tensor of shape (num_pointers, 768)
                            Point3R's internal decoder features
                            Only present if the model returns this field
            - 'camera_poses': (Optional) Tensor of shape (num_frames, 7)
                             Camera poses for each frame in format [tx, ty, tz, qw, qx, qy, qz]
                             Only present if the Point3R model has pose_head=True
                             Translation: [tx, ty, tz] - absolute position in 3D space
                             Rotation: [qw, qx, qy, qz] - unit quaternion (real part first)
                             Coordinate convention: OpenCV camera-to-world transformation

    Example:
        >>> from qwen_vl_utils import process_vision_info
        >>> messages = [{"role": "user", "content": [{"type": "image", "image": "path/to/image.jpg"}]}]
        >>> image_inputs, _ = process_vision_info(messages)
        >>>
        >>> pointer_data = extract_pointer_memory(
        ...     image_inputs,
        ...     point3r_model,
        ...     device='cuda'
        ... )
        >>>
        >>> # Use with Qwen2_5_VLForConditionalGenerationWithPoint3R
        >>> inputs = processor(
        ...     text=["<|pointer_pad|> What's in this scene?"],
        ...     return_tensors="pt",
        ... )
        >>> outputs = model.generate(
        ...     **inputs,
        ...     pointer_memory_embeds=pointer_data['pointer_memory_embeds'],
        ...     pointer_positions=pointer_data['pointer_positions'],
        ... )
    """

    # Ensure image_inputs is a list
    if not isinstance(image_inputs, list):
        image_inputs = [image_inputs]

    # Prepare images for Point3R using the simplified function
    # Determine target size based on the size parameter
    if isinstance(size, tuple):
        target_size = size
    elif size == 512:
        target_size = (640, 480)  # Default for size=512
    elif size == 224:
        target_size = (224, 224)
    else:
        # For other sizes, maintain 4:3 aspect ratio
        target_size = (size, int(size * 3 / 4))

    crop_border = 0 if no_crop else 20
    views = prepare_images_for_point3r(
        image_inputs,
        target_size=target_size,
        crop_border=crop_border
    )

    if verbose:
        for i, view in enumerate(views):
            print(f"Processed image {i+1}/{len(image_inputs)}: shape {view['true_shape']}")

    # Run Point3R inference
    if verbose:
        print(f"Running Point3R inference on {len(views)} image(s)...")

    outputs = inference(
        views,
        point3r_model,
        device,
        image_embeds=image_embeds,
        grid_thw=grid_thw,
        deepstack_image_embeds=deepstack_image_embeds,
        verbose=verbose,
        lambda_decay=lambda_decay,
        max_memory_tokens=max_memory_tokens,
    )
        
    # Extract memory_aligned_image_embeds from Point3R outputs
    if 'memory_aligned_image_embeds' in outputs and outputs['memory_aligned_image_embeds'] is not None:
        memory_aligned_image_embeds = outputs['memory_aligned_image_embeds']
        if isinstance(memory_aligned_image_embeds, list):
            memory_aligned_image_embeds = memory_aligned_image_embeds[-1]
        if memory_aligned_image_embeds.dim() == 3:
            memory_aligned_image_embeds = memory_aligned_image_embeds[0]
        pointer_memory_embeds = memory_aligned_image_embeds
        if verbose:
            print(f"Extracted memory_aligned_image_embeds: {memory_aligned_image_embeds.shape}")
    else:
        raise ValueError("memory_aligned_image_embeds not found in outputs")

    # Extract memory_feat from Point3R outputs
    memory_feat = None
    if 'memory_feat' in outputs and outputs['memory_feat'] is not None:
        memory_feat = outputs['memory_feat']
        if isinstance(memory_feat, list):
            memory_feat = memory_feat[-1]
        if memory_feat.dim() == 3:
            memory_feat = memory_feat[0]
        if verbose:
            print(f"Extracted memory_feat: {memory_feat.shape}")

    # Extract deepstack_memory_aligned_embeds from Point3R outputs
    deepstack_memory_aligned_embeds = None
    if 'deepstack_memory_aligned_embeds' in outputs and outputs['deepstack_memory_aligned_embeds'] is not None:
        deepstack_memory_aligned_embeds = outputs['deepstack_memory_aligned_embeds']
        # deepstack is a list of per-layer embeddings, each is a list (per batch) or tensor
        processed_deepstack = []
        for layer_embeds in deepstack_memory_aligned_embeds:
            if isinstance(layer_embeds, list):
                layer_embeds = layer_embeds[-1]  # Take last batch element
            if layer_embeds.dim() == 3:
                layer_embeds = layer_embeds[0]  # Remove batch dimension
            processed_deepstack.append(layer_embeds)
        deepstack_memory_aligned_embeds = processed_deepstack
        if verbose:
            print(f"Extracted deepstack_memory_aligned_embeds: {len(deepstack_memory_aligned_embeds)} layers")
            for i, layer in enumerate(deepstack_memory_aligned_embeds):
                print(f"  - Layer {i}: {layer.shape}")

    # Extract memory_aligned_timestamps from Point3R outputs
    pointer_timestamps = None
    if 'memory_aligned_timestamps' in outputs and outputs['memory_aligned_timestamps'] is not None:
        memory_aligned_timestamps = outputs['memory_aligned_timestamps']
        if isinstance(memory_aligned_timestamps, list):
            memory_aligned_timestamps = memory_aligned_timestamps[-1]  # Take last batch element
        if memory_aligned_timestamps.dim() == 2:
            memory_aligned_timestamps = memory_aligned_timestamps[0]  # Remove batch dimension
        pointer_timestamps = memory_aligned_timestamps.cpu()
        if verbose:
            print(f"Extracted memory_aligned_timestamps: {pointer_timestamps.shape[0]} tokens")
            print(f"  - Timestamp range: [{pointer_timestamps.min().item()}, {pointer_timestamps.max().item()}]")
            print(f"  - Unique timestamps: {pointer_timestamps.unique().tolist()}")

    # Extract pos_decode_memory from Point3R outputs
    if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
        pos_decode_memory = outputs['pos_decode_memory']
        if isinstance(pos_decode_memory, list):
            pos_decode_memory = pos_decode_memory[-1]
        if pos_decode_memory.dim() == 3:
            pos_decode_memory = pos_decode_memory[0]

        pointer_positions = pos_decode_memory.cpu()
        if verbose:
            print(f"Extracted pos_decode_memory: {pointer_positions.shape[0]} points")
            print(f"  - xyz ranges: x[{pointer_positions[:, 0].min():.3f}, {pointer_positions[:, 0].max():.3f}], "
                  f"y[{pointer_positions[:, 1].min():.3f}, {pointer_positions[:, 1].max():.3f}], "
                  f"z[{pointer_positions[:, 2].min():.3f}, {pointer_positions[:, 2].max():.3f}]")

    # Sort all pointer data by ascending timestamp
    if pointer_timestamps is not None:
        sort_indices = torch.argsort(pointer_timestamps)
        pointer_timestamps = pointer_timestamps[sort_indices]
        pointer_memory_embeds = pointer_memory_embeds[sort_indices]
        pointer_positions = pointer_positions[sort_indices]
        if memory_feat is not None:
            memory_feat = memory_feat[sort_indices]
        if deepstack_memory_aligned_embeds is not None:
            deepstack_memory_aligned_embeds = [layer[sort_indices] for layer in deepstack_memory_aligned_embeds]
        if verbose:
            print(f"Sorted pointer data by timestamp (ascending)")

    # Extract camera poses from Point3R predictions (if pose_head=True)
    camera_poses = []
    if 'pred' in outputs and outputs['pred'] is not None:
        for i, pred in enumerate(outputs['pred']):
            if 'camera_pose' in pred and pred['camera_pose'] is not None:
                # camera_pose shape: (batch_size, 7) where 7 = [tx, ty, tz, qw, qx, qy, qz]
                pose = pred['camera_pose']
                if pose.dim() == 2:
                    # Take first batch element if batched
                    pose = pose[0]  # Shape: (7,)
                camera_poses.append(pose.cpu())
        if len(camera_poses) > 0:
            # Stack all camera poses: (num_frames, 7)
            camera_poses = torch.stack(camera_poses, dim=0)
            if verbose:
                print(f"Extracted camera poses:")
                print(f"  - Number of frames: {camera_poses.shape[0]}")
                print(f"  - Pose format: [tx, ty, tz, qw, qx, qy, qz]")
                print(f"  - Translation ranges: x[{camera_poses[:, 0].min():.3f}, {camera_poses[:, 0].max():.3f}], "
                      f"y[{camera_poses[:, 1].min():.3f}, {camera_poses[:, 1].max():.3f}], "
                      f"z[{camera_poses[:, 2].min():.3f}, {camera_poses[:, 2].max():.3f}]")
        else:
            camera_poses = None
            if verbose:
                print(f"No camera poses found (pose_head may be disabled)")
    else:
        camera_poses = None

    if verbose:
        print(f"Extracted pointer memory:")
        print(f"  - Number of pointers: {pointer_memory_embeds.shape[0]}")
        print(f"  - Memory embeddings shape: {pointer_memory_embeds.shape}")
        print(f"  - Pointer positions shape: {pointer_positions.shape}")
        if memory_feat is not None:
            print(f"  - Memory feat shape: {memory_feat.shape}")
        if 'pos_decode_memory' in outputs and outputs['pos_decode_memory'] is not None:
            print(f"  - Final position ranges: x[{pointer_positions[:, 0].min():.3f}, {pointer_positions[:, 0].max():.3f}], "
                  f"y[{pointer_positions[:, 1].min():.3f}, {pointer_positions[:, 1].max():.3f}], "
                  f"z[{pointer_positions[:, 2].min():.3f}, {pointer_positions[:, 2].max():.3f}]")

    result = {
        'pointer_memory_embeds': pointer_memory_embeds,
        'pointer_positions': pointer_positions,
        '_point3r_outputs': outputs,
    }

    # Add memory_feat if available
    if memory_feat is not None:
        result['memory_feat'] = memory_feat

    # Add camera poses if available
    if camera_poses is not None:
        result['camera_poses'] = camera_poses

    # Add deepstack_image_embeds if available
    if deepstack_memory_aligned_embeds is not None:
        result['deepstack_image_embeds'] = deepstack_memory_aligned_embeds

    # Add pointer_timestamps if available
    if pointer_timestamps is not None:
        result['pointer_timestamps'] = pointer_timestamps

    # Add frames_indices if available
    if frames_indices is not None:
        result['frames_indices'] = frames_indices

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Online / VLA inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def init_online_memory(point3r_model):
    """Return an empty MemoryState dict for online (VLA) inference.

    Call this once before the frame loop, then pass the returned dict to
    ``step_online_memory()`` at each step.

    Args:
        point3r_model: Initialised ``Point3R`` model instance.

    Returns:
        dict — empty memory state with ``frame_idx=0``.
    """
    return point3r_model.init_memory_state()


def _extract_pointer_data_from_state(memory_state):
    """Extract a ``pointer_data`` dict from a MemoryState produced by
    ``step_online_memory()``.

    Mirrors the extraction logic at the tail of ``extract_pointer_memory``
    (lines 584–734), but reads directly from the in-memory state dict
    instead of re-running Point3R inference.

    Args:
        memory_state: dict returned by ``step_online_memory()``.

    Returns:
        dict with keys ``pointer_memory_embeds``, ``pointer_positions``,
        ``pointer_timestamps``, and optionally ``deepstack_image_embeds``.
        Ready to pass to ``run_models(pointer_data=...)``.
    """
    # ── pointer_memory_embeds ─────────────────────────────────────────────────
    mae = memory_state['memory_aligned_image_embeds']
    if mae is None:
        raise ValueError("memory_aligned_image_embeds is None — has step_online_memory been called?")
    # After i>=1, _forward_addmemory_merge returns a list (one element per batch)
    if isinstance(mae, list):
        pointer_memory_embeds = mae[-1]
    elif mae.dim() == 3:
        pointer_memory_embeds = mae[0]   # remove batch dim
    else:
        pointer_memory_embeds = mae

    # ── pointer_positions ─────────────────────────────────────────────────────
    pdm = memory_state['pos_decode_memory']
    if pdm is None:
        raise ValueError("pos_decode_memory is None")
    if isinstance(pdm, list):
        pointer_positions = pdm[-1].cpu()
    elif pdm.dim() == 3:
        pointer_positions = pdm[0].cpu()
    else:
        pointer_positions = pdm.cpu()

    # ── pointer_timestamps ────────────────────────────────────────────────────
    pts = memory_state['memory_aligned_timestamps']
    pointer_timestamps = None
    if pts is not None:
        if isinstance(pts, list):
            pointer_timestamps = pts[-1].cpu()
        elif pts.dim() == 2:
            pointer_timestamps = pts[0].cpu()
        else:
            pointer_timestamps = pts.cpu()

    # ── Sort by ascending timestamp ───────────────────────────────────────────
    if pointer_timestamps is not None:
        sort_idx              = torch.argsort(pointer_timestamps)
        pointer_timestamps    = pointer_timestamps[sort_idx]
        pointer_memory_embeds = pointer_memory_embeds[sort_idx]
        pointer_positions     = pointer_positions[sort_idx]

    result = {
        'pointer_memory_embeds': pointer_memory_embeds,
        'pointer_positions':     pointer_positions,
    }
    if pointer_timestamps is not None:
        result['pointer_timestamps'] = pointer_timestamps

    # ── deepstack_image_embeds ────────────────────────────────────────────────
    ds = memory_state['deepstack_memory_aligned_embeds']
    if ds is not None:
        processed = []
        for layer_embeds in ds:
            # Each layer may be a list (one per batch) or a tensor
            if isinstance(layer_embeds, list):
                le = layer_embeds[-1]
            elif layer_embeds.dim() == 3:
                le = layer_embeds[0]
            else:
                le = layer_embeds
            if pointer_timestamps is not None:
                le = le[sort_idx]
            processed.append(le)
        result['deepstack_image_embeds'] = processed

    return result


def _extract_visual_embeddings_batched(model, processor, image_inputs, min_pixels, max_pixels, batch_size=2):
    """Extract image embeddings from the visual encoder in batches.
    Handles Qwen3-VL (tuple output with deepstack), Qwen3.5 (BaseModelOutputWithPooling), and Qwen2.5-VL (raw tensor).
    """
    import torch
    _visual = model.visual if hasattr(model, 'visual') else model.model.visual
    model_device = next(_visual.parameters()).device
    image_embeds_list = []
    deepstack_image_embeds = []
    grid_thw_list = []

    for i in range(0, len(image_inputs), batch_size):
        batch_images = image_inputs[i:i+batch_size]
        processed_batch = processor.image_processor(images=batch_images, min_pixels=min_pixels, max_pixels=max_pixels)

        with torch.inference_mode():
            pixel_values = processed_batch.pixel_values.type(_visual.dtype).to(model_device)
            grid_thw = processed_batch.image_grid_thw

            visual_output = _visual(pixel_values, grid_thw=grid_thw)
            batch_deepstack_image_embeds = None
            if isinstance(visual_output, tuple):
                batch_image_embeds, batch_deepstack_image_embeds = visual_output
            elif hasattr(visual_output, 'pooler_output'):
                batch_image_embeds = visual_output.pooler_output
            else:
                batch_image_embeds = visual_output

            image_embeds_list.append(batch_image_embeds)
            if batch_deepstack_image_embeds is not None:
                if len(deepstack_image_embeds) == 0:
                    deepstack_image_embeds = [[] for _ in range(len(batch_deepstack_image_embeds))]
                for layer_idx, layer_embeds in enumerate(batch_deepstack_image_embeds):
                    deepstack_image_embeds[layer_idx].append(layer_embeds)
            grid_thw_list.append(grid_thw)

    image_embeds = torch.cat(image_embeds_list, dim=0)
    grid_thw = torch.cat(grid_thw_list, dim=0)
    if deepstack_image_embeds:
        deepstack_image_embeds = [torch.cat(layer_embeds_list, dim=0) for layer_embeds_list in deepstack_image_embeds]

    return image_embeds, grid_thw, deepstack_image_embeds


def step_online_memory(
    new_frame,
    memory_state,
    point3r_model,
    model,
    processor,
    min_pixels,
    max_pixels,
    max_memory_tokens=512,
    lambda_decay=1.0,
):
    """Process ONE new frame, update the Point3R memory, and return
    a ``pointer_data`` dict ready for ``run_models()``.

    No disk I/O.  Memory lives entirely on GPU between calls.

    Args:
        new_frame: PIL Image of the new observation.
        memory_state: dict from ``init_online_memory()`` or a previous
                      ``step_online_memory()`` call.
        point3r_model: Initialised ``Point3R`` model.
        model: Qwen LLM (provides the visual encoder).
        processor: Qwen processor (provides image_processor).
        min_pixels: Passed to the image processor.
        max_pixels: Passed to the image processor.
        max_memory_tokens: Hard cap on memory token count after each step.
        lambda_decay: EMA decay for merged Qwen embeddings.

    Returns:
        pointer_data (dict): pass to ``run_models(pointer_data=...)``.
        new_memory_state (dict): pass to the next ``step_online_memory()`` call.
    """
    import torch

    point3r_device = next(point3r_model.parameters()).device

    # ── 1. Extract Qwen visual embeddings for this single frame ───────────────
    image_embeds, grid_thw, deepstack_image_embeds = _extract_visual_embeddings_batched(
        model, processor, [new_frame], min_pixels, max_pixels, batch_size=1
    )
    image_embeds = image_embeds.to(point3r_device)
    grid_thw     = grid_thw.to(point3r_device)
    if deepstack_image_embeds:
        deepstack_image_embeds = [l.to(point3r_device) for l in deepstack_image_embeds]

    # ── 2. Prepare the Point3R view dict ─────────────────────────────────────
    t, h, w      = grid_thw[0].tolist()
    merge_size   = processor.image_processor.merge_size
    patch_size   = processor.image_processor.patch_size
    target_size  = (
        (w // merge_size) * patch_size,   # width
        (h // merge_size) * patch_size,   # height
    )
    views = prepare_images_for_point3r([new_frame], target_size=target_size, crop_border=0)
    view  = {k: v.to(point3r_device) for k, v in views[0].items()}

    # ── 3. Incremental Point3R step ───────────────────────────────────────────
    new_memory_state = point3r_model.step_online(
        view,
        memory_state,
        image_embeds_i=image_embeds,
        grid_thw_i=grid_thw,
        deepstack_embeds_i=deepstack_image_embeds if deepstack_image_embeds else None,
        max_memory_tokens=max_memory_tokens,
        lambda_decay=lambda_decay,
    )

    # ── 4. Extract pointer_data from updated state ────────────────────────────
    pointer_data = _extract_pointer_data_from_state(new_memory_state)
    return pointer_data, new_memory_state


if __name__ == "__main__":
    # Example usage
    print("Example: Extract pointer memory from an image")
    print("=" * 70)

    # This is a demonstration - you would provide actual images
    print("Usage:")
    print("""
    from extract_pointer_memory import extract_pointer_memory
    from qwen_vl.model.point3r.point3r import Point3R

    # Load Point3R model
    point3r_model = Point3R.from_pretrained("path/to/point3r_checkpoint.pth")
    point3r_model = point3r_model.to('cuda')
    point3r_model.eval()

    # Extract memory from images
    pointer_data = extract_pointer_memory(
        image_inputs=['path/to/image1.jpg', 'path/to/image2.jpg'],
        point3r_model=point3r_model,
        device='cuda',
        no_crop=False,
        size=512,
    )

    # Use with Qwen2.5-VL model
    outputs = model.generate(
        **inputs,
        pointer_memory_embeds=pointer_data['pointer_memory_embeds'],
        pointer_positions=pointer_data['pointer_positions'],
    )

    # Access camera poses (if pose_head=True)
    if 'camera_poses' in pointer_data:
        camera_poses = pointer_data['camera_poses']  # Shape: (num_frames, 7)
        # Each pose: [tx, ty, tz, qw, qx, qy, qz]

        # Convert to 4x4 camera-to-world matrices
        from src.qwen_vl.model.point3r.utils.camera import pose_encoding_to_camera
        c2w_matrices = pose_encoding_to_camera(camera_poses, pose_encoding_type='absT_quaR')
        # Shape: (num_frames, 4, 4) - OpenCV camera-to-world transformations
    """)
