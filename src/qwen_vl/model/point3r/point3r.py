import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from .utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from .heads import head_factory
from .utils.camera import PoseEncoder
from .patch_embed import get_patch_embed
from ..croco.croco import CroCoNet, CrocoConfig
from .point3r_blocks import (
    Block,
    MemoryDecoderBlock,
    DecoderBlock,
    PosDecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """
    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None
    memory_aligned_image_embeds: Optional[torch.Tensor] = None
    pos_decode_memory: Optional[torch.Tensor] = None
    memory_feat: Optional[torch.Tensor] = None
    deepstack_memory_aligned_embeds: Optional[List[torch.Tensor]] = None
    memory_aligned_timestamps: Optional[torch.Tensor] = None

def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def from_dust3r_to_ours(state_dict):
    
    new_state_dict = OrderedDict()
        
    for k, v in state_dict.items():
        if k.startswith("dec_blocks2."):
            k = k.replace("dec_blocks2.", "dec_blocks_memory.")
        elif k.startswith("downstream_head1.dpt."):
            k = k.replace("downstream_head1.dpt.", "downstream_head.dpt_self.")
        elif k.startswith("downstream_head2.dpt."):
            k = k.replace("downstream_head2.dpt.", "downstream_head.dpt_cross.")
        name = k
        new_state_dict[name] = v
        
    return new_state_dict

def load_model(model_path, device):
    
    print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]["model"].replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")  
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    assert "landscape_only=False" in args
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    print(s)
    return net.to(device)


class Point3RConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="dpt",
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        local_mem_size=256,
        memory_dec_num_heads=16,
        depth_head=False,
        pose_conf_head=False,
        pose_head=False,
        use_merge=True,
        merge_threshold=None,
        len_unit=20,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.memory_dec_num_heads = memory_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.use_merge = use_merge
        self.merge_threshold = merge_threshold
        self.len_unit = len_unit
        self.croco_kwargs = croco_kwargs

# thanks to CUT3R (https://github.com/CUT3R)
class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                PosDecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                PosDecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query) 
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]


class Point3R(CroCoNet):
    config_class = Point3RConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: Point3RConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        self.use_merge = config.use_merge
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True)
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,)
        
        self._set_memory_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.memory_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )

        self._set_value_encoder(
            enc_depth=6, 
            enc_embed_dim=1024, 
            out_dim=1024, 
            enc_num_heads=16,
            mlp_ratio=4, 
            norm_layer=self.croco_args.get("norm_layer", None),
        )

        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.memory_attn_head = nn.Sequential(
            nn.Linear(self.enc_embed_dim+self.dec_embed_dim, self.enc_embed_dim+self.dec_embed_dim),
            nn.GELU(),
            nn.Linear(self.enc_embed_dim+self.dec_embed_dim, self.enc_embed_dim))

        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(Point3R, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.pts_patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    rope3d=self.rope3d,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_memory_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_memory = dec_depth
        self.dec_embed_dim_memory = dec_embed_dim
        self.decoder_embed_memory = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_memory = nn.ModuleList(
            [
                MemoryDecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                    rope3d=self.rope3d,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_memory = norm_layer(dec_embed_dim)
    
    def _set_value_encoder(
        self,
        enc_depth, 
        enc_embed_dim, 
        out_dim, 
        enc_num_heads,
        mlp_ratio, 
        norm_layer
    ):
        self.value_encoder = nn.ModuleList(
            [
                Block(
                    enc_embed_dim, 
                    enc_num_heads, 
                    mlp_ratio, 
                    qkv_bias=True, 
                    norm_layer=norm_layer, 
                    rope=self.rope
                )
                for i in range(enc_depth)
            ]
        )
        self.value_norm = norm_layer(enc_embed_dim)
        self.value_out = nn.Linear(enc_embed_dim, out_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_memory") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_memory")] = value
        if not any(k.startswith("pts_patch_embed") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("patch_embed"):
                    new_ckpt[key.replace("patch_embed", "pts_patch_embed")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze): 
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "encoder": [
                self.patch_embed,
                self.enc_blocks,
                self.enc_norm,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape, sub_batch_size=64):
        n = image.shape[0]
        if n <= sub_batch_size:
            x, pos = self.patch_embed(image, true_shape=true_shape)
            assert self.enc_pos_embed is None
            for blk in self.enc_blocks:
                x = blk(x, pos)
            x = self.enc_norm(x)
            return [x], pos, None

        # Process in sub-batches to avoid OOM on large image counts
        all_x, all_pos = [], []
        for i in range(0, n, sub_batch_size):
            chunk_img = image[i:i + sub_batch_size]
            chunk_shape = true_shape[i:i + sub_batch_size]
            x, pos = self.patch_embed(chunk_img, true_shape=chunk_shape)
            assert self.enc_pos_embed is None
            for blk in self.enc_blocks:
                x = blk(x, pos)
            x = self.enc_norm(x)
            all_x.append(x)
            all_pos.append(pos)
        return [torch.cat(all_x, dim=0)], torch.cat(all_pos, dim=0), None

    def _encode_views(self, views, img_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        img_mask = torch.stack(
            [view["img_mask"] for view in views], dim=0
        ) 
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
        full_pos[img_masks_flat] += img_pos
        
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, i, mask_memory, f_memory, pos_memory, f_img, pos_img, f_pose, point3r_tag=False):
        if isinstance(f_memory, torch.Tensor):
            assert f_memory.shape[-1] == self.dec_embed_dim
        else:
            assert f_memory[-1].shape[-1] == self.dec_embed_dim
        
        final_output = [(f_memory, f_img)] 
        f_img = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1)
        final_output.append((f_memory, f_img))
        
        for blk_memory, blk_img in zip(self.dec_blocks_memory, self.dec_blocks):
            f_memory, _ = blk_memory(i, *final_output[-1][::+1], mask_memory, pos_memory, pos_img, point3r_tag=point3r_tag)
            f_img, _ = blk_img(i, *final_output[-1][::-1], mask_memory, pos_img, pos_memory, point3r_tag=point3r_tag)
            final_output.append((f_memory, f_img))
        del final_output[1] 
        final_output[-1] = (
            self.dec_norm_memory(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_memory(self, image_tokens, image_pos):
        
        memory_feat = self.decoder_embed_memory(image_tokens)
        return memory_feat, None

    def _recurrent_rollout(
        self,
        i,
        mask_memory,
        memory_feat,
        memory_pos,
        current_feat,
        current_pos,
        pose_feat,
        point3r_tag=False,
    ):
        new_memory_feat, dec = self._decoder(
            i,
            mask_memory,
            memory_feat, memory_pos, current_feat, current_pos, pose_feat,
            point3r_tag=point3r_tag,
        )
        new_memory_feat = new_memory_feat[-1]
        return new_memory_feat, dec

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)
  
    def enc_pts_value(self, pts, shape):
        out, pos = self.pts_patch_embed(pts.permute(0, 3, 1, 2), true_shape=shape)
        for block in self.value_encoder:
            out = block(out, pos)
        out = self.value_norm(out)
        out = self.value_out(out)
        return out

    def _interpolate_image_embeds_to_point3r_grid(
        self,
        image_embeds,     # (num_patches_qwen, embed_dim) - Qwen's embeddings after spatial merge
        grid_thw,         # (num_images, 3) - [T, H, W] before merge
        target_shape      # (H//16, W//16) - Point3R's 16x16 grid
    ):
        """
        Interpolate Qwen's image embeddings to match Point3R's patch grid.

        Qwen3-VL: patch_size=16, spatial_merge_size=2 -> effective stride 32px.
        Qwen2.5-VL: patch_size=14, spatial_merge_size=2 -> effective stride 28px.
        Point3R: patch_size=16 -> stride 16px.

        We bilinearly interpolate from Qwen's merged grid to Point3R's target grid.
        When all images share the same grid dimensions (common case), this is done
        as a single batched operation for efficiency.

        Args:
            image_embeds: (num_patches_total, embed_dim) concatenated across images
            grid_thw: (num_images, 3) containing [temporal, height, width] BEFORE merge
            target_shape: (target_h, target_w) - Point3R's expected grid size

        Returns:
            interpolated: (bs, target_h * target_w, embed_dim) aligned with Point3R patches
        """
        import torch.nn.functional as F

        bs = grid_thw.shape[0]
        embed_dim = image_embeds.shape[-1]
        target_h, target_w = target_shape
        spatial_merge_size = 2

        # Compute post-merge grid dimensions per image
        h_merged = grid_thw[:, 1] // spatial_merge_size
        w_merged = grid_thw[:, 2] // spatial_merge_size
        
        assert target_h == h_merged
        assert target_w == w_merged

        # Fast path: all images share the same grid dimensions (common case)
        if (h_merged == h_merged[0]).all() and (w_merged == w_merged[0]).all():
            h_qwen = h_merged[0].item()
            w_qwen = w_merged[0].item()

            # Reshape all images at once: (bs*h*w, C) -> (bs, h, w, C) -> (bs, C, h, w)
            embeds_spatial = image_embeds.reshape(bs, h_qwen, w_qwen, embed_dim).permute(0, 3, 1, 2)

            # Chunk to avoid exceeding INT_MAX elements in F.interpolate
            max_batch = max(1, (2**31 - 1) // (embed_dim * target_h * target_w))
            if bs <= max_batch:
                interpolated_spatial = F.interpolate(
                    embeds_spatial,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                chunks = [
                    F.interpolate(chunk, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    for chunk in embeds_spatial.split(max_batch, dim=0)
                ]
                interpolated_spatial = torch.cat(chunks, dim=0)

            # (bs, C, th, tw) -> (bs, th*tw, C)
            return interpolated_spatial.reshape(bs, embed_dim, -1).permute(0, 2, 1)

        # Slow path: mixed resolutions - process per-image
        patches_per_image = (h_merged * w_merged).tolist()
        image_embeds_list = torch.split(image_embeds, patches_per_image, dim=0)

        interpolated_list = []
        for i, img_embeds in enumerate(image_embeds_list):
            h_q = h_merged[i].item()
            w_q = w_merged[i].item()

            # (h*w, C) -> (1, h, w, C) -> (1, C, h, w)
            embeds_spatial = img_embeds.reshape(1, h_q, w_q, embed_dim).permute(0, 3, 1, 2)

            interpolated_spatial = F.interpolate(
                embeds_spatial,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            )

            # (1, C, th, tw) -> (th*tw, C)
            interpolated_list.append(
                interpolated_spatial.reshape(embed_dim, -1).permute(1, 0)
            )

        return torch.stack(interpolated_list, dim=0)

    def _forward_addmemory(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        memory_aligned_image_embeds=None,
        image_embeds_i=None,
        deepstack_memory_aligned_embeds=None,
        deepstack_image_embeds_i=None,
        memory_aligned_timestamps=None,
        lambda_decay=1.0,
        max_memory_tokens=None,
        ):
        bs, img_h, img_w, _ = pts3d.shape
        img_pos_len_h = img_h // 16
        img_pos_len_w = img_w // 16
        img_pos = pts3d.permute(0, 3, 1, 2)
        img_pos = img_pos.unfold(2, 16, 16)
        img_pos = img_pos.unfold(3, 16, 16)
        img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)

        feat_key = self.memory_attn_head(torch.cat((feat_i, dec_i), dim=-1))
        feat_pts = self.enc_pts_value(pts3d, shape_i)
        memory_add = self.decoder_embed_memory(feat_key+feat_pts)
        memory_add = memory_add.float()

        if i == 0:
            memory_feat = memory_add
            init_memory_feat = memory_feat.clone().detach()
            chosen_pts = img_pos
            # Initialize image embeds, deepstack, and timestamps
            memory_aligned_image_embeds = image_embeds_i
            deepstack_memory_aligned_embeds = deepstack_image_embeds_i
            num_tokens = memory_add.shape[1]
            memory_aligned_timestamps = torch.zeros(bs, num_tokens, dtype=torch.long, device=memory_add.device)
        else:
            memory_feat = torch.cat((memory_feat, memory_add), dim=1)
            init_memory_feat = torch.cat((init_memory_feat, memory_add.clone().detach()), dim=1)
            chosen_pts = torch.cat((memory_pos, img_pos), dim=1)
            # Concatenate image embeds
            if image_embeds_i is not None and memory_aligned_image_embeds is not None:
                memory_aligned_image_embeds = torch.cat((memory_aligned_image_embeds, image_embeds_i), dim=1)
            # Concatenate deepstack per layer
            if deepstack_image_embeds_i is not None and deepstack_memory_aligned_embeds is not None:
                deepstack_memory_aligned_embeds = [
                    torch.cat((existing, new), dim=1)
                    for existing, new in zip(deepstack_memory_aligned_embeds, deepstack_image_embeds_i)
                ]
            # Concatenate timestamps
            if memory_aligned_timestamps is not None:
                num_new = memory_add.shape[1]
                new_timestamps = torch.full((bs, num_new), i, dtype=torch.long, device=memory_add.device)
                memory_aligned_timestamps = torch.cat((memory_aligned_timestamps, new_timestamps), dim=1)

            # Prune if over budget
            if max_memory_tokens is not None and memory_feat.shape[1] > max_memory_tokens:
                perm = torch.randperm(memory_feat.shape[1], device=memory_feat.device)
                keep = perm[:max_memory_tokens]
                keep, _ = torch.sort(keep)
                memory_feat = memory_feat[:, keep]
                init_memory_feat = init_memory_feat[:, keep]
                chosen_pts = chosen_pts[:, keep]
                if memory_aligned_image_embeds is not None:
                    memory_aligned_image_embeds = memory_aligned_image_embeds[:, keep]
                if deepstack_memory_aligned_embeds is not None:
                    deepstack_memory_aligned_embeds = [layer[:, keep] for layer in deepstack_memory_aligned_embeds]
                if memory_aligned_timestamps is not None:
                    memory_aligned_timestamps = memory_aligned_timestamps[:, keep]

        return memory_feat, chosen_pts, init_memory_feat, img_pos, \
               memory_aligned_image_embeds, deepstack_memory_aligned_embeds, memory_aligned_timestamps

    def _forward_addmemory_merge(
        self,
        i,
        pts3d,
        init_memory_feat,
        memory_feat,
        memory_pos,
        feat_i,
        dec_i,
        shape_i,
        memory_aligned_image_embeds=None,  # Accumulated image embeddings aligned with memory tokens
        image_embeds_i=None,  # New image embeddings for current view
        deepstack_memory_aligned_embeds=None,  # Accumulated deepstack embeddings (list of per-layer embeds)
        deepstack_image_embeds_i=None,  # New deepstack embeddings for current view (list of per-layer embeds)
        memory_aligned_timestamps=None,  # Accumulated timestamps for each memory token
        lambda_decay=1.0,  # EMA decay factor: updated = lambda * new + (1 - lambda) * old
        max_memory_tokens=None,  # Maximum number of memory tokens before pruning
        ):
        """
        Add memory tokens with spatial merging to avoid duplicate points.

        When i==0: Initializes memory from first view
        When i>0: Merges new points with existing memory based on spatial proximity:
            - Points far from existing memory → added as new tokens
            - Points close to existing memory → merged by averaging

        Args:
            i: View index
            pts3d: 3D point cloud from current view [bs, img_h, img_w, 3]
            memory_feat: Current memory features (tensor if i<2, list if i>=2)
            memory_pos: 3D positions of memory tokens
            memory_aligned_image_embeds: Accumulated Qwen image embeddings aligned with memory
            image_embeds_i: New Qwen image embeddings for current view [bs, num_patches, embed_dim]
            deepstack_memory_aligned_embeds: Accumulated deepstack embeddings (list of per-layer embeds)
            deepstack_image_embeds_i: New deepstack embeddings for current view (list of per-layer embeds)
            memory_aligned_timestamps: Accumulated timestamps for each memory token (frame index when last updated)

        Returns:
            memory_feat: Updated memory features
            chosen_pts/memory_pos: Updated 3D positions
            init_memory_feat: Snapshot of memory (unused, kept for compatibility)
            img_pos: Downsampled 3D positions from current view
            memory_aligned_image_embeds: Updated image embeddings aligned with memory
            deepstack_memory_aligned_embeds: Updated deepstack embeddings aligned with memory
            memory_aligned_timestamps: Updated timestamps for each memory token
        """
        bs, img_h, img_w, _ = pts3d.shape
        img_pos_len_h = img_h // 16
        img_pos_len_w = img_w // 16
        img_pos = pts3d.permute(0, 3, 1, 2)
        img_pos = img_pos.unfold(2, 16, 16)
        img_pos = img_pos.unfold(3, 16, 16)
        img_pos = img_pos.reshape(bs, 3, img_pos_len_h, img_pos_len_w, -1).mean(dim=-1).permute(0, 2, 3, 1).reshape(bs, -1, 3)

        feat_key = self.memory_attn_head(torch.cat((feat_i, dec_i), dim=-1))
        feat_pts = self.enc_pts_value(pts3d, shape_i)
        memory_add = self.decoder_embed_memory(feat_key+feat_pts)
        memory_add = memory_add.float()

        if i == 0:
            memory_feat = memory_add
            init_memory_feat = memory_feat.clone().detach()
            chosen_pts = img_pos
            # NEW: Initialize memory_aligned_image_embeds
            memory_aligned_image_embeds = image_embeds_i
            # NEW: Initialize deepstack_memory_aligned_embeds
            deepstack_memory_aligned_embeds = deepstack_image_embeds_i
            # NEW: Initialize memory_aligned_timestamps to 0 for all tokens
            num_tokens = memory_add.shape[1]  # (bs, num_tokens, dim)
            memory_aligned_timestamps = torch.zeros(bs, num_tokens, dtype=torch.long, device=memory_add.device)
        else:
            memory_feat_list = []
            memory_pos_list = []
            # NEW: Track pointer_image_embeds in parallel
            image_embeds_list = []
            # NEW: Track deepstack embeddings per layer
            takes_deepstack = deepstack_image_embeds_i is not None and deepstack_memory_aligned_embeds is not None
            num_deepstack_layers = len(deepstack_image_embeds_i) if takes_deepstack else 0
            deepstack_embeds_lists = [[] for _ in range(num_deepstack_layers)]  # List of lists, one per layer
            # NEW: Track timestamps per batch
            timestamps_list = []

            takes_image_embeds = image_embeds_i is not None and memory_aligned_image_embeds is not None

            for j in range(bs):
                memory_pos_j = memory_pos[j]
                memory_feat_j = memory_feat[j]
                img_pos_j = img_pos[j]
                new_feat_j = memory_add[j]

                # Extract image embeddings for batch j
                # image_embeds_j: existing accumulated embeddings for this batch
                # new_image_embeds_j: new embeddings from current view for this batch
                if takes_image_embeds:
                    image_embeds_j = memory_aligned_image_embeds[j]
                    new_image_embeds_j = image_embeds_i[j]

                # NEW: Extract deepstack embeddings for batch j (per layer)
                if takes_deepstack:
                    deepstack_embeds_j = [layer_embeds[j] for layer_embeds in deepstack_memory_aligned_embeds]
                    new_deepstack_embeds_j = [layer_embeds[j] for layer_embeds in deepstack_image_embeds_i]

                # NEW: Extract timestamps for batch j
                timestamps_j = memory_aligned_timestamps[j]

                # Compute threshold based on point cloud extent
                if self.config.merge_threshold is not None:
                    threshold_j = self.config.merge_threshold
                else:
                    _len_unit = self.config.len_unit
                    unit_j = (torch.cat((memory_pos_j, img_pos_j), dim=0).max(dim=0).values - torch.cat((memory_pos_j, img_pos_j), dim=0).min(dim=0).values) / _len_unit
                    threshold_j = torch.norm(unit_j)
                distances = torch.cdist(img_pos_j, memory_pos_j)
                min_dists, min_indices = distances.min(dim=-1)
                mask_add = min_dists >= threshold_j
                mask_merge = ~mask_add
                if mask_merge.sum() > 0:
                    indices_merge = min_indices[mask_merge]
                    pos_merge = img_pos_j[mask_merge]
                    feat_merge = new_feat_j[mask_merge]
                    unique_indices, inverse_indices = torch.unique(indices_merge, return_inverse=True)
                    num_merge = unique_indices.shape[0]
                    pos_sum = torch.zeros((num_merge, 3), device=memory_pos_j.device)
                    feat_sum = torch.zeros((num_merge, 768), device=memory_feat_j.device)
                    count = torch.zeros((num_merge, 1), device=memory_feat_j.device)
                    pos_sum.index_add_(0, inverse_indices, pos_merge)
                    feat_sum.index_add_(0, inverse_indices, feat_merge)
                    count.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32).unsqueeze(1))
                    pos_avg = pos_sum / count
                    pos_avg = pos_avg.float()
                    feat_avg = feat_sum / count
                    feat_avg = feat_avg.float()
                    memory_pos_j[unique_indices] = pos_avg
                    memory_feat_j[unique_indices] = feat_avg

                    # NEW: Apply same merge logic to image embeddings (at embed_dim, dynamically determined)
                    if takes_image_embeds:
                        image_feat_merge = new_image_embeds_j[mask_merge]
                        embed_dim_image = image_embeds_j.shape[-1]  # Dynamically get dimension from input
                        image_feat_sum = torch.zeros((num_merge, embed_dim_image), device=image_embeds_j.device,
                                                     dtype=image_feat_merge.dtype)
                        image_feat_sum.index_add_(0, inverse_indices, image_feat_merge)
                        image_feat_avg = image_feat_sum / count
                        image_feat_avg = image_feat_avg.bfloat16()
                        image_embeds_j[unique_indices] = lambda_decay * image_feat_avg + (1 - lambda_decay) * image_embeds_j[unique_indices]

                    # NEW: Apply same merge logic to deepstack embeddings (per layer)
                    if takes_deepstack:
                        for layer_idx in range(num_deepstack_layers):
                            ds_feat_merge = new_deepstack_embeds_j[layer_idx][mask_merge]
                            embed_dim_ds = deepstack_embeds_j[layer_idx].shape[-1]
                            ds_feat_sum = torch.zeros((num_merge, embed_dim_ds), device=deepstack_embeds_j[layer_idx].device,
                                                      dtype=ds_feat_merge.dtype)
                            ds_feat_sum.index_add_(0, inverse_indices, ds_feat_merge)
                            ds_feat_avg = ds_feat_sum / count
                            ds_feat_avg = ds_feat_avg.bfloat16()
                            deepstack_embeds_j[layer_idx][unique_indices] = lambda_decay * ds_feat_avg + (1 - lambda_decay) * deepstack_embeds_j[layer_idx][unique_indices]

                    # NEW: Update timestamps for merged tokens to current frame index
                    timestamps_j[unique_indices] = i

                if mask_add.sum() > 0:
                    pos_add = img_pos_j[mask_add]
                    feat_add = new_feat_j[mask_add]
                    memory_pos_j = torch.cat([memory_pos_j, pos_add], dim=0)
                    memory_feat_j = torch.cat([memory_feat_j, feat_add], dim=0)

                    # NEW: Add new image embeddings at new locations
                    if takes_image_embeds:
                        image_feat_add = new_image_embeds_j[mask_add]
                        image_embeds_j = torch.cat([image_embeds_j, image_feat_add], dim=0)

                    # NEW: Add new deepstack embeddings at new locations (per layer)
                    if takes_deepstack:
                        for layer_idx in range(num_deepstack_layers):
                            ds_feat_add = new_deepstack_embeds_j[layer_idx][mask_add]
                            deepstack_embeds_j[layer_idx] = torch.cat([deepstack_embeds_j[layer_idx], ds_feat_add], dim=0)

                    # NEW: Add timestamps for new tokens (set to current frame index)
                    num_new_tokens = mask_add.sum().item()
                    new_timestamps = torch.full((num_new_tokens,), i, dtype=torch.long, device=timestamps_j.device)
                    timestamps_j = torch.cat([timestamps_j, new_timestamps], dim=0)

                # Prune memory tokens if over budget (random selection)
                if max_memory_tokens is not None and memory_feat_j.shape[0] > max_memory_tokens:
                    perm = torch.randperm(memory_feat_j.shape[0], device=memory_feat_j.device)
                    keep_indices = perm[:max_memory_tokens]
                    keep_indices, _ = torch.sort(keep_indices)
                    memory_feat_j = memory_feat_j[keep_indices]
                    memory_pos_j = memory_pos_j[keep_indices]
                    timestamps_j = timestamps_j[keep_indices]
                    if takes_image_embeds:
                        image_embeds_j = image_embeds_j[keep_indices]
                    if takes_deepstack:
                        for layer_idx in range(num_deepstack_layers):
                            deepstack_embeds_j[layer_idx] = deepstack_embeds_j[layer_idx][keep_indices]

                memory_feat_list.append(memory_feat_j)
                memory_pos_list.append(memory_pos_j)
                # NEW: Track image embeddings
                if takes_image_embeds:
                    image_embeds_list.append(image_embeds_j)
                # NEW: Track deepstack embeddings per layer
                if takes_deepstack:
                    for layer_idx in range(num_deepstack_layers):
                        deepstack_embeds_lists[layer_idx].append(deepstack_embeds_j[layer_idx])
                # NEW: Track timestamps
                timestamps_list.append(timestamps_j)
            init_memory_feat_list = [memory_feat_index.clone().detach() for memory_feat_index in memory_feat_list]

        if i == 0:
            return memory_feat, chosen_pts, init_memory_feat, img_pos, memory_aligned_image_embeds, deepstack_memory_aligned_embeds, memory_aligned_timestamps
        else:
            # Note: deepstack_embeds_lists is a list of lists: [[batch0_layer0, batch1_layer0, ...], [batch0_layer1, ...], ...]
            # This matches the structure expected when indexing by layer then by batch
            return memory_feat_list, memory_pos_list, init_memory_feat_list, img_pos, image_embeds_list, deepstack_embeds_lists if takes_deepstack else None, timestamps_list

    def _forward_merge(self, views, point3r_tag=False, image_embeds=None, grid_thw_images=None, deepstack_image_embeds=None, lambda_decay=1.0, max_memory_tokens=None):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_memory_feat = memory_feat.clone()
        ress = []
        pos_decode_img = None
        pos_decode_memory = None
        merge_tag = False
        # NEW: Initialize memory_aligned_image_embeds
        memory_aligned_image_embeds = None
        # NEW: Initialize deepstack_memory_aligned_embeds (list of per-layer accumulated embeddings)
        deepstack_memory_aligned_embeds = None
        # NEW: Initialize memory_aligned_timestamps
        memory_aligned_timestamps = None
        # Prepare per-view image embeds for lazy interpolation (saves GPU memory)
        _interp_target_shape = None
        _image_embeds_split = None
        _deepstack_splits = None
        if image_embeds is not None and grid_thw_images is not None:
            img_h, img_w = shape[0][0].tolist()  # Get height and width from first view's shape
            img_pos_len_h = img_h // 16
            img_pos_len_w = img_w // 16
            _interp_target_shape = (img_pos_len_h, img_pos_len_w)
            # Split concatenated embeddings into per-image tensors (views, no copy)
            spatial_merge_size = 2
            h_merged = grid_thw_images[:, 1] // spatial_merge_size
            w_merged = grid_thw_images[:, 2] // spatial_merge_size
            patches_per_image = (h_merged * w_merged).tolist()
            _image_embeds_split = list(torch.split(image_embeds, [int(p) for p in patches_per_image], dim=0))

            if deepstack_image_embeds is not None:
                _deepstack_splits = [
                    list(torch.split(layer_embeds, [int(p) for p in patches_per_image], dim=0))
                    for layer_embeds in deepstack_image_embeds
                ]

        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            # Interpolate current view's embeddings on-the-fly to save GPU memory
            if _image_embeds_split is not None:
                img_emb_i = self._interpolate_image_embeds_to_point3r_grid(
                    _image_embeds_split[i],
                    grid_thw_images[i:i+1],
                    target_shape=_interp_target_shape
                )  # (1, target_h*target_w, embed_dim)
            else:
                img_emb_i = None
            deepstack_emb_i = None
            if _deepstack_splits is not None:
                deepstack_emb_i = [
                    self._interpolate_image_embeds_to_point3r_grid(
                        _deepstack_splits[layer_idx][i],
                        grid_thw_images[i:i+1],
                        target_shape=_interp_target_shape
                    )
                    for layer_idx in range(len(_deepstack_splits))
                ]
            if i >= 2:
                merge_tag = True
            if merge_tag:
                memory_len_max = max(f_memory_j.shape[0] for f_memory_j in memory_feat)
                f_memory_list_padded = []
                pos_memory_list_padded = []
                mask_memory_list_padded = []
                for j in range(len(memory_feat)):
                    f_memory_j = memory_feat[j]
                    pos_memory_j = pos_decode_memory[j]
                    padding_size = memory_len_max - f_memory_j.shape[0]
                    padding = torch.zeros(padding_size, f_memory_j.shape[1]).to(f_memory_j.device)
                    padding_pos = torch.zeros(padding_size, pos_memory_j.shape[1]).to(pos_memory_j.device)
                    mask_valid = torch.ones(f_memory_j.shape[0]).to(f_memory_j.device)
                    mask_invalid = torch.zeros(padding_size).to(f_memory_j.device)
                    padded_memory_j = torch.cat((f_memory_j, padding), dim=0)
                    padded_pos_memory_j = torch.cat((pos_memory_j, padding_pos), dim=0)
                    padded_mask_j = torch.cat((mask_valid, mask_invalid), dim=0)
                    f_memory_list_padded.append(padded_memory_j)
                    pos_memory_list_padded.append(padded_pos_memory_j)
                    mask_memory_list_padded.append(padded_mask_j)
                memory_feat = torch.stack(f_memory_list_padded, dim=0)
                pos_decode_memory = torch.stack(pos_memory_list_padded, dim=0)
                mask_memory = torch.stack(mask_memory_list_padded, dim=0)
            else:
                mask_memory = None
            
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            else:
                pose_feat_i = None

            new_memory_feat, dec = self._recurrent_rollout(
                i,
                mask_memory,
                memory_feat,
                pos_decode_memory,
                feat_i,
                pos_decode_img,
                pose_feat_i,
                point3r_tag=point3r_tag,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]

            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)  
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)
            if mask_memory is not None:
                memory_feat_new_list = []
                pos_decode_memory_new_list = []
                for j in range(mask_memory.shape[0]):
                    j_mask_memory = mask_memory[j]
                    j_mask_memory = j_mask_memory.bool()
                    j_memory_feat = memory_feat[j]
                    j_pos_decode_memory = pos_decode_memory[j]
                    j_memory_feat = j_memory_feat[j_mask_memory]
                    j_pos_decode_memory = j_pos_decode_memory[j_mask_memory]
                    memory_feat_new_list.append(j_memory_feat)
                    pos_decode_memory_new_list.append(j_pos_decode_memory)
                memory_feat = memory_feat_new_list
                pos_decode_memory = pos_decode_memory_new_list
                init_memory_feat = [memory_feat_in.clone().detach() for memory_feat_in in memory_feat]

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
                if pos_decode_memory is not None:
                    if isinstance(pos_decode_memory, torch.Tensor):
                        pos_decode_memory = pos_decode_memory.clone().detach()
                    else:
                        pos_decode_memory = [pos_decode_memory_in.clone().detach() for pos_decode_memory_in in pos_decode_memory]
                memory_feat, pos_decode_memory, init_memory_feat, pos_decode_img, memory_aligned_image_embeds, deepstack_memory_aligned_embeds, memory_aligned_timestamps = self._forward_addmemory_merge(
                    i,
                    pts3d=this_pts3d,
                    init_memory_feat=init_memory_feat,
                    memory_feat=memory_feat,
                    memory_pos=pos_decode_memory,
                    feat_i=feat_i.clone().detach(),
                    dec_i=dec[-1][:, 1:].clone().detach(),
                    shape_i=views[i]['true_shape'],
                    memory_aligned_image_embeds=memory_aligned_image_embeds,
                    image_embeds_i=img_emb_i,
                    deepstack_memory_aligned_embeds=deepstack_memory_aligned_embeds,
                    deepstack_image_embeds_i=deepstack_emb_i,
                    memory_aligned_timestamps=memory_aligned_timestamps,
                    lambda_decay=lambda_decay,
                    max_memory_tokens=max_memory_tokens,
                )

            # Free intermediate tensors to reduce GPU memory pressure
            del dec, head_input, new_memory_feat
            if i % 50 == 0 and i > 0:
                torch.cuda.empty_cache()

        return ress, views, memory_aligned_image_embeds, pos_decode_memory, memory_feat, deepstack_memory_aligned_embeds, memory_aligned_timestamps

    def _forward(self, views, point3r_tag=False, image_embeds=None, grid_thw_images=None, deepstack_image_embeds=None, lambda_decay=1.0, max_memory_tokens=None):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        memory_feat, _ = self._init_memory(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_memory_feat = memory_feat.clone()
        ress = []
        pos_decode_img = None
        pos_decode_memory = None
        memory_aligned_image_embeds = None
        deepstack_memory_aligned_embeds = None
        memory_aligned_timestamps = None

        # Prepare per-view image embeds for lazy interpolation
        _interp_target_shape = None
        _image_embeds_split = None
        _deepstack_splits = None
        if image_embeds is not None and grid_thw_images is not None:
            img_h, img_w = shape[0][0].tolist()
            img_pos_len_h = img_h // 16
            img_pos_len_w = img_w // 16
            _interp_target_shape = (img_pos_len_h, img_pos_len_w)
            spatial_merge_size = 2
            h_merged = grid_thw_images[:, 1] // spatial_merge_size
            w_merged = grid_thw_images[:, 2] // spatial_merge_size
            patches_per_image = (h_merged * w_merged).tolist()
            _image_embeds_split = list(torch.split(image_embeds, [int(p) for p in patches_per_image], dim=0))
            if deepstack_image_embeds is not None:
                _deepstack_splits = [
                    list(torch.split(layer_embeds, [int(p) for p in patches_per_image], dim=0))
                    for layer_embeds in deepstack_image_embeds
                ]

        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]

            # Interpolate current view's embeddings on-the-fly
            if _image_embeds_split is not None:
                img_emb_i = self._interpolate_image_embeds_to_point3r_grid(
                    _image_embeds_split[i],
                    grid_thw_images[i:i+1],
                    target_shape=_interp_target_shape
                )
            else:
                img_emb_i = None
            deepstack_emb_i = None
            if _deepstack_splits is not None:
                deepstack_emb_i = [
                    self._interpolate_image_embeds_to_point3r_grid(
                        _deepstack_splits[layer_idx][i],
                        grid_thw_images[i:i+1],
                        target_shape=_interp_target_shape
                    )
                    for layer_idx in range(len(_deepstack_splits))
                ]

            mask_memory = None

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            else:
                pose_feat_i = None
            new_memory_feat, dec = self._recurrent_rollout(
                i,
                mask_memory,
                memory_feat,
                pos_decode_memory,
                feat_i,
                pos_decode_img,
                pose_feat_i,
                point3r_tag=point3r_tag,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)

            update_mask_memory = torch.tensor([False]*memory_feat.shape[0], device=memory_feat.device)
            update_mask_memory = update_mask_memory[:, None, None].float()
            memory_feat = new_memory_feat * update_mask_memory + memory_feat * (1 - update_mask_memory)
            update_mask_mem = torch.tensor([True]*mem.shape[0], device=mem.device)
            update_mask_mem = update_mask_mem[:, None, None].float()
            mem = new_mem * update_mask_mem + mem * (1 - update_mask_mem)

            if point3r_tag:
                this_pts3d = res['pts3d_in_other_view'].clone().detach()
                if pos_decode_memory is not None:
                    if isinstance(pos_decode_memory, torch.Tensor):
                        pos_decode_memory = pos_decode_memory.clone().detach()
                    else:
                        pos_decode_memory = [pos_decode_memory_in.clone().detach() for pos_decode_memory_in in pos_decode_memory]
                memory_feat, pos_decode_memory, init_memory_feat, pos_decode_img, \
                    memory_aligned_image_embeds, deepstack_memory_aligned_embeds, \
                    memory_aligned_timestamps = self._forward_addmemory(
                    i,
                    pts3d=this_pts3d,
                    init_memory_feat=init_memory_feat,
                    memory_feat=memory_feat,
                    memory_pos=pos_decode_memory,
                    feat_i=feat_i.clone().detach(),
                    dec_i=dec[-1][:, 1:].clone().detach(),
                    shape_i=views[i]['true_shape'],
                    memory_aligned_image_embeds=memory_aligned_image_embeds,
                    image_embeds_i=img_emb_i,
                    deepstack_memory_aligned_embeds=deepstack_memory_aligned_embeds,
                    deepstack_image_embeds_i=deepstack_emb_i,
                    memory_aligned_timestamps=memory_aligned_timestamps,
                    lambda_decay=lambda_decay,
                    max_memory_tokens=max_memory_tokens,
                )

            # Free intermediate tensors to reduce GPU memory pressure
            del dec, head_input, new_memory_feat
            if i % 50 == 0 and i > 0:
                torch.cuda.empty_cache()

        return ress, views, memory_aligned_image_embeds, pos_decode_memory, memory_feat, deepstack_memory_aligned_embeds, memory_aligned_timestamps

    def forward(self, views, point3r_tag=False, image_embeds=None, grid_thw_images=None, deepstack_image_embeds=None, lambda_decay=1.0, max_memory_tokens=None):
        forward_fn = self._forward_merge if self.use_merge else self._forward
        ress, views, memory_aligned_image_embeds, pos_decode_memory, memory_feat, deepstack_memory_aligned_embeds, memory_aligned_timestamps = forward_fn(
            views,
            point3r_tag=point3r_tag,
            image_embeds=image_embeds,
            grid_thw_images=grid_thw_images,
            deepstack_image_embeds=deepstack_image_embeds,
            lambda_decay=lambda_decay,
            max_memory_tokens=max_memory_tokens,
        )
        return ARCroco3DStereoOutput(
            ress=ress,
            views=views,
            memory_aligned_image_embeds=memory_aligned_image_embeds,
            pos_decode_memory=pos_decode_memory,
            memory_feat=memory_feat,
            deepstack_memory_aligned_embeds=deepstack_memory_aligned_embeds,
            memory_aligned_timestamps=memory_aligned_timestamps
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Online / VLA inference API
    # ──────────────────────────────────────────────────────────────────────────

    def init_memory_state(self):
        """Return an empty MemoryState dict for online (VLA) inference.

        Call once before the frame loop, then pass the returned dict to
        ``step_online()`` at each step.
        """
        return {
            'memory_feat':                    None,
            'pos_decode_memory':              None,
            'mem':                            None,
            'init_memory_feat':               None,
            'memory_aligned_image_embeds':    None,
            'deepstack_memory_aligned_embeds': None,
            'memory_aligned_timestamps':      None,
            'frame_idx':                      0,
        }

    @torch.no_grad()
    def step_online(
        self,
        view,
        memory_state,
        image_embeds_i=None,
        grid_thw_i=None,
        deepstack_embeds_i=None,
        max_memory_tokens=512,
        lambda_decay=1.0,
    ):
        """Process ONE new frame and return an updated MemoryState.

        Mirrors a single iteration of ``_forward_merge`` but accepts and
        returns an explicit state dict, enabling incremental / online
        inference without disk I/O.  Designed for batch_size=1 (VLA).

        Args:
            view: dict with ``'img'`` (1,3,H,W), ``'true_shape'`` (1,2),
                  ``'img_mask'`` (1,) on the target device.
            memory_state: dict from ``init_memory_state()`` or a previous
                          ``step_online()`` call.
            image_embeds_i: (N_patches, embed_dim) Qwen visual embeddings for
                            this frame, or None.
            grid_thw_i: (1, 3) grid [T, H, W] before spatial merge, or None.
            deepstack_embeds_i: list of (N_patches, d_layer) per-layer Qwen
                                embeddings, or None.
            max_memory_tokens: hard cap on accumulated memory tokens.
            lambda_decay: EMA decay for merged Qwen embeddings.

        Returns:
            new_memory_state (dict) — pass directly to the next call.
        """
        i      = memory_state['frame_idx']
        device = view['img'].device

        # ── 1. Encode single frame ────────────────────────────────────────────
        shape, feat_ls, pos = self._encode_views([view])
        feat_i = feat_ls[-1][0]  # (1, N_patches, D)
        pos_i  = pos[0]          # (1, N_patches, D_pos)

        # ── 2. Interpolate Qwen embeddings to Point3R's 16×16 grid ───────────
        img_emb_i       = None
        deepstack_emb_i = None
        if image_embeds_i is not None and grid_thw_i is not None:
            img_h, img_w = shape[0][0].tolist()
            target_shape = (img_h // 16, img_w // 16)
            img_emb_i = self._interpolate_image_embeds_to_point3r_grid(
                image_embeds_i, grid_thw_i, target_shape=target_shape
            )  # (1, target_h*target_w, embed_dim)
            if deepstack_embeds_i is not None:
                deepstack_emb_i = [
                    self._interpolate_image_embeds_to_point3r_grid(
                        layer_emb, grid_thw_i, target_shape=target_shape
                    )
                    for layer_emb in deepstack_embeds_i
                ]

        # ── 3. Unpack persisted state ─────────────────────────────────────────
        memory_feat                     = memory_state['memory_feat']
        pos_decode_memory               = memory_state['pos_decode_memory']
        mem                             = memory_state['mem']
        init_memory_feat                = memory_state['init_memory_feat']
        memory_aligned_image_embeds     = memory_state['memory_aligned_image_embeds']
        deepstack_memory_aligned_embeds = memory_state['deepstack_memory_aligned_embeds']
        memory_aligned_timestamps       = memory_state['memory_aligned_timestamps']

        # ── 4. Initialise pose-retriever mem on first frame ───────────────────
        if mem is None:
            mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1).clone()

        # ── 5. Initialise spatial memory on first frame ───────────────────────
        if memory_feat is None:
            memory_feat, _ = self._init_memory(feat_i, pos_i)
            init_memory_feat = memory_feat.clone()

        # ── 6. Prepare decoder inputs ─────────────────────────────────────────
        #   i <  2: memory_feat is a tensor — pass directly, no padding needed.
        #   i >= 2: memory_feat is a list of per-batch tensors (variable length
        #           after spatial merging); pad to uniform length for the decoder.
        merge_tag = (i >= 2)
        if merge_tag:
            memory_len_max = max(f.shape[0] for f in memory_feat)
            f_padded_list, p_padded_list, mask_list = [], [], []
            for j in range(len(memory_feat)):
                f_j  = memory_feat[j]
                p_j  = pos_decode_memory[j]
                pad  = memory_len_max - f_j.shape[0]
                f_padded = torch.cat([f_j, torch.zeros(pad, f_j.shape[1], device=device)], dim=0) if pad > 0 else f_j
                p_padded = torch.cat([p_j, torch.zeros(pad, p_j.shape[1], device=device)], dim=0) if pad > 0 else p_j
                mask     = torch.cat([torch.ones(f_j.shape[0], device=device),
                                      torch.zeros(pad, device=device)])
                f_padded_list.append(f_padded)
                p_padded_list.append(p_padded)
                mask_list.append(mask)
            memory_feat_tensor    = torch.stack(f_padded_list, dim=0)
            pos_decode_mem_tensor = torch.stack(p_padded_list, dim=0)
            mask_memory           = torch.stack(mask_list, dim=0)
        else:
            memory_feat_tensor    = memory_feat          # tensor
            pos_decode_mem_tensor = pos_decode_memory    # None (i==0) or tensor (i==1)
            mask_memory           = None

        # ── 7. Pose retriever ─────────────────────────────────────────────────
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
        else:
            global_img_feat_i = None
            pose_feat_i       = None

        # ── 8. Decoder rollout ────────────────────────────────────────────────
        new_memory_feat, dec = self._recurrent_rollout(
            i, mask_memory, memory_feat_tensor, pos_decode_mem_tensor,
            feat_i, None, pose_feat_i, point3r_tag=True,
        )

        # ── 9. Update pose-retriever state ────────────────────────────────────
        if self.pose_head_flag:
            out_pose_feat_i = dec[-1][:, 0:1]
            mem = self.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)

        # ── 10. Downstream head → pts3d ───────────────────────────────────────
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res        = self._downstream_head(head_input, shape[0], pos=pos_i)
        this_pts3d = res['pts3d_in_other_view'].clone().detach()

        # ── 11. Restore list form before _forward_addmemory_merge ────────────
        #   (must match what the previous call returned)
        if merge_tag:
            memory_feat_for_add = []
            pos_for_add         = []
            for j in range(mask_memory.shape[0]):
                m = mask_memory[j].bool()
                memory_feat_for_add.append(memory_feat_tensor[j][m])
                pos_for_add.append(pos_decode_mem_tensor[j][m])
            init_memory_feat = [f.clone().detach() for f in memory_feat_for_add]
        else:
            memory_feat_for_add = memory_feat_tensor
            if pos_decode_memory is not None and isinstance(pos_decode_memory, torch.Tensor):
                pos_for_add = pos_decode_memory.clone().detach()
            else:
                pos_for_add = pos_decode_memory

        # ── 12. Add / merge new frame into persistent memory ──────────────────
        (
            memory_feat_new,
            pos_decode_memory_new,
            init_memory_feat_new,
            _,
            memory_aligned_image_embeds_new,
            deepstack_memory_aligned_embeds_new,
            memory_aligned_timestamps_new,
        ) = self._forward_addmemory_merge(
            i,
            pts3d=this_pts3d,
            init_memory_feat=init_memory_feat,
            memory_feat=memory_feat_for_add,
            memory_pos=pos_for_add,
            feat_i=feat_i.clone().detach(),
            dec_i=dec[-1][:, 1:].clone().detach(),
            shape_i=view['true_shape'],
            memory_aligned_image_embeds=memory_aligned_image_embeds,
            image_embeds_i=img_emb_i,
            deepstack_memory_aligned_embeds=deepstack_memory_aligned_embeds,
            deepstack_image_embeds_i=deepstack_emb_i,
            memory_aligned_timestamps=memory_aligned_timestamps,
            lambda_decay=lambda_decay,
            max_memory_tokens=max_memory_tokens,
        )

        del dec, head_input, new_memory_feat
        if i % 50 == 0 and i > 0:
            torch.cuda.empty_cache()

        # ── 13. Pack and return updated state ─────────────────────────────────
        return {
            'memory_feat':                    memory_feat_new,
            'pos_decode_memory':              pos_decode_memory_new,
            'mem':                            mem,
            'init_memory_feat':               init_memory_feat_new,
            'memory_aligned_image_embeds':    memory_aligned_image_embeds_new,
            'deepstack_memory_aligned_embeds': deepstack_memory_aligned_embeds_new,
            'memory_aligned_timestamps':      memory_aligned_timestamps_new,
            'frame_idx':                      i + 1,
        }

    