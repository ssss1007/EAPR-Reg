import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
from .modelio import LoadableModel, store_config_args
import math

# from . import layers


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # print('size', size)

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        # new locations
        # print('flow', flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvResBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = x + out
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=8):
        super(Encoder, self).__init__()
        c = first_channel
        self.block1 = ConvBlock(in_channel, c)
        self.block2 = ConvBlock(c, c * 2)
        self.block3 = ConvBlock(c *2, c * 4)
        self.block4 = ConvBlock(c *4, c * 4)

    def forward(self, x):
        out1 = self.block1(x)
        x = nn.AvgPool3d(2)(out1)
        out2 = self.block2(x)
        x = nn.AvgPool3d(2)(out2)
        out3 = self.block3(x)
        x = nn.AvgPool3d(2)(out3)
        out4 = self.block4(x)
        return out1, out2, out3, out4

class DecoderBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.Conv1 = ConvBlock(x_channel+y_channel, out_channel)
        self.Conv2 = ConvResBlock(out_channel, out_channel)
        self.Conv3 = ConvResBlock(out_channel, out_channel)
        # self.Conv3_2 = ConvResBlock(out_channel, out_channel)
        # self.Conv3_3 = ConvResBlock(out_channel, out_channel)
        self.Conv4 = nn.Conv3d(out_channel, out_channel//2, 3, padding=1)
        self.Conv5 = nn.Conv3d(out_channel//2, 3, 3, padding=1)
        nn.init.normal_(self.Conv5.weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.Conv5.bias, 0.0)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        cost_vol = self.Conv1(concat)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        # cost_vol = self.Conv3_2(cost_vol)
        # cost_vol = self.Conv3_3(cost_vol)
        cost_vol = self.Conv4(cost_vol)
        flow = self.Conv5(cost_vol)

        return flow


class FineRefineEncoder(nn.Module):
    """
    Lightweight feature extractor for fine refinement stage.
    """

    def __init__(self, in_channel=1, out_channel=32):
        super().__init__()
        mid = max(8, out_channel // 2)
        self.net = nn.Sequential(
            nn.Conv3d(in_channel, mid, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(mid, out_channel, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)



class ChannelAdapterAPE(nn.Module):
    def __init__(self):
        super().__init__()
        # 抛弃 1x1 卷积，我们直接用物理空间相邻的切片来填满 3 个通道

    def forward(self, x):
        """
        x: [B, 1, D, H, W] (单模态 3D 图像，例如 T1 MRI)
        """
        B, C, D, H, W = x.shape
        assert C == 1, f"Expected 1 channel, but got {C}"

        # 沿着 Depth 维度 (第 2 维) 在前后各 padding 1 层
        # pad 格式是从后往前填的：(W左, W右, H上, H下, D前, D后)
        # 使用 'replicate' 边缘复制模式，比填 0 更合理，避免头尾切片产生纯黑的伪影
        x_padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode='replicate')

        # 提取前一帧 (z-1), 当前帧 (z), 后一帧 (z+1)
        # 此时 x_padded 的 depth 长度为 D + 2
        prev_slice = x_padded[:, :, 0:D, :, :]      # [B, 1, D, H, W]
        curr_slice = x_padded[:, :, 1:D+1, :, :]    # [B, 1, D, H, W]
        next_slice = x_padded[:, :, 2:D+2, :, :]    # [B, 1, D, H, W]

        # 在 Channel 维度拼接，完美构成 3 通道 "RGB-like" 输入
        rgb_like = torch.cat([prev_slice, curr_slice, next_slice], dim=1) 
        
        return rgb_like  # [B, 3, D, H, W]

class MedSAM2Encoder3D(nn.Module):
    """
    Frozen MedSAM2 encoder used to generate 3D multi-scale features.
    Note: the slice traversal order matches the original APE pretraining code.
    """

    def __init__(self, cfg_path, ckpt_path, sam2_repo=None, device='cuda'):
        super().__init__()

        try:
            from ixi__eapr.sam2_runtime import build_sam2_from_paths
        except ModuleNotFoundError:
            # Direct invocation: python ixi__eapr/train.py (ixi__eapr is on sys.path).
            from sam2_runtime import build_sam2_from_paths
        model = build_sam2_from_paths(cfg_path, ckpt_path, repo=sam2_repo, device=device)
        self.backbone = model.image_encoder
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.adapter = ChannelAdapterAPE()

    @torch.no_grad()
    def forward(self, volume, chunk_size=16):
        x = self.adapter(volume)
        B, C, D, H, W = x.shape

        x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        feats_3d = [[], [], []]
        for start in range(0, B * D, chunk_size):
            end = min(start + chunk_size, B * D)
            out = self.backbone(x_2d[start:end])
            fpn = out["backbone_fpn"]
            if len(fpn) != 3:
                raise RuntimeError(f"Expected 3 FPN levels, got {len(fpn)}")
            for i in range(3):
                feats_3d[i].append(fpn[i])

        feats_all = [torch.cat(stage_list, dim=0) for stage_list in feats_3d]

        feats_3d_dhw = []
        for f in feats_all:
            _, C2, H2, W2 = f.shape
            f = f.view(B, D, C2, H2, W2).permute(0, 2, 1, 3, 4).contiguous()
            feats_3d_dhw.append(f)

        return feats_3d_dhw


class APEGuidanceHead(nn.Module):
    """APE head used for extracting prototype assignment maps."""

    def __init__(self, k_list=(24, 10, 6), temp=0.3, channels=64):
        super().__init__()
        self.in_channels = [256, 256, 256]
        if isinstance(channels, (list, tuple)):
            if len(channels) != len(self.in_channels):
                raise ValueError("APE channels list must have three entries.")
            self.channels = [int(c) for c in channels]
        else:
            self.channels = [int(channels)] * len(self.in_channels)
        self.K = [int(k) for k in k_list]
        self.register_buffer("current_temp", torch.tensor(float(temp)))

        self.projectors = nn.ModuleList()
        for in_c, out_c in zip(self.in_channels, self.channels):
            self.projectors.append(
                nn.Sequential(
                    nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                    nn.InstanceNorm3d(out_c, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        self.protos = nn.ParameterList(
            [nn.Parameter(torch.randn(k, c) * 0.02) for c, k in zip(self.channels, self.K)]
        )

        self.cs_convs = nn.ModuleList()
        for l in range(len(self.K) - 1):
            in_ch = self.K[l + 1]
            out_ch = self.K[l]
            conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)
            nn.init.constant_(conv.weight, 1.0 / in_ch)
            nn.init.zeros_(conv.bias)
            self.cs_convs.append(conv)

    def forward(self, feats_list):
        proj_feats = []
        for proj, feat in zip(self.projectors, feats_list):
            proj_feats.append(proj(feat))

        R_list, A_list = [], []
        for i, (feat, proto) in enumerate(zip(proj_feats, self.protos)):
            B, C, D, H, W = feat.shape
            feat_flat = feat.view(B, C, -1).transpose(1, 2)
            feat_norm = nnf.normalize(feat_flat, dim=-1)
            proto_norm = nnf.normalize(proto, dim=-1)

            sim = torch.einsum("bnc,kc->bnk", feat_norm, proto_norm) / self.current_temp
            A = nnf.softmax(sim, dim=-1)

            A_vol = A.view(B, D, H, W, self.K[i]).permute(0, 4, 1, 2, 3)

            smooth_ratio = 0.2 if i == 0 else 0.3
            A_blur = F.avg_pool3d(A_vol, kernel_size=3, stride=1, padding=1)
            A_vol = (1.0 - smooth_ratio) * A_vol + smooth_ratio * A_blur
            A_vol = A_vol / (A_vol.sum(dim=1, keepdim=True) + 1e-8)

            A_smooth = A_vol.permute(0, 2, 3, 4, 1).reshape(B, -1, self.K[i])
            R = torch.einsum("bnk,kc->bnc", A_smooth, proto)
            R = R.transpose(1, 2).view(B, C, D, H, W)

            R_list.append(R)
            A_list.append(A_vol)

        return R_list, A_list, proj_feats


class RawMedSAMProjector(nn.Module):
    """Project raw MedSAM2 FPN features to the same 64-channel space as APE."""

    def __init__(self):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(256, 64, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(64, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for _ in range(3)
        ])

    def forward(self, feats_list):
        return [proj(feat) for proj, feat in zip(self.projectors, feats_list)]


def _infer_ape_channels_from_state_dict(state_dict, default=64):
    """Infer APE feature width from a checkpoint state dict."""
    for key, value in state_dict.items():
        if key.endswith("projectors.0.0.weight") and value.ndim >= 1:
            return int(value.shape[0])
        if key.endswith("protos.0") and value.ndim == 2:
            return int(value.shape[1])
    return int(default)


class PyramidFeatureFusion(nn.Module):
    """Fuse APE prototype features with projected raw MedSAM2 features."""

    def __init__(self, channels=64):
        super().__init__()
        self.fusers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for _ in range(3)
        ])

    def forward(self, ape_feats, raw_feats):
        return [
            fuser(torch.cat([ape_feat, raw_feat], dim=1))
            for fuser, ape_feat, raw_feat in zip(self.fusers, ape_feats, raw_feats)
        ]


class PyramidGatedResidualFusion(nn.Module):
    """Gated residual fusion that keeps APE features as the main branch."""

    def __init__(self, channels=64):
        super().__init__()
        self.gates = nn.ModuleList()
        for _ in range(3):
            gate = nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)
            self.gates.append(gate)

    def forward(self, ape_feats, raw_feats):
        fused = []
        for gate, ape_feat, raw_feat in zip(self.gates, ape_feats, raw_feats):
            g = torch.sigmoid(gate(torch.cat([ape_feat, raw_feat], dim=1)))
            fused.append(ape_feat + g * raw_feat)
        return fused


def _normalized_assignment_entropy(assignments, eps=1e-8):
    """Compute voxel-wise normalized entropy from soft prototype assignments."""
    entropy = -(assignments * torch.log(assignments.clamp_min(eps))).sum(dim=1, keepdim=True)
    denom = math.log(max(2, int(assignments.shape[1])))
    return entropy / denom


class EntropyAwareDetailFusion(nn.Module):
    """Fuse APE semantic anchors with projected detail features using assignment entropy."""

    def __init__(self, channels=64):
        super().__init__()
        self.detail_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for _ in range(3)
        ])
        self.gates = nn.ModuleList([
            nn.Conv3d(channels * 2 + 1, channels, kernel_size=3, padding=1, bias=True)
            for _ in range(3)
        ])

    def forward(self, ape_feats, assignment_maps, projected_feats):
        fused = []
        for detail_block, gate, ape_feat, assign_map, proj_feat in zip(
            self.detail_blocks,
            self.gates,
            ape_feats,
            assignment_maps,
            projected_feats,
        ):
            uncertainty = _normalized_assignment_entropy(assign_map).to(dtype=ape_feat.dtype)
            detail = detail_block(torch.cat([proj_feat, proj_feat - ape_feat], dim=1))
            gate_map = torch.sigmoid(gate(torch.cat([ape_feat, proj_feat, uncertainty], dim=1)))
            fused.append(ape_feat + gate_map * detail)
        return fused


class PrototypeTokenAttention(nn.Module):
    """Lightweight voxel-to-prototype attention for APE feature enhancement."""

    def __init__(self, channels=64, num_heads=4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat, proto_tokens):
        """
        feat: [B, C, D, H, W]
        proto_tokens: [T, C], prototype tokens from one or more APE levels.
        """
        B, C, D, H, W = feat.shape
        tokens = proto_tokens.to(device=feat.device, dtype=feat.dtype)

        q = self.q_proj(feat).view(B, self.num_heads, self.head_dim, -1)
        q = q.permute(0, 1, 3, 2).contiguous()  # [B, heads, N, head_dim]

        k = self.k_proj(tokens).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(tokens).view(-1, self.num_heads, self.head_dim)
        k = k.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
        v = v.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 1, 3, 2).contiguous().view(B, C, D, H, W)
        return feat + self.gamma * self.out_proj(ctx)


class AnatomicalPrototypeAttention(nn.ModuleList):
    """Multi-scale anatomical prototype attention used by APAR-Net."""

    def __init__(self, channels=64, num_heads=4):
        super().__init__([
            PrototypeTokenAttention(channels=channels, num_heads=num_heads)
            for _ in range(3)
        ])

    def forward(self, feat_list, proto_list):
        """
        feat_list: three APE feature maps ordered from fine to coarse.
        proto_list: three anatomical prototype banks with matching order.
        """
        if len(feat_list) != 3 or len(proto_list) != 3:
            raise ValueError("AnatomicalPrototypeAttention expects three feature/prototype levels.")

        f1, f2, f3 = feat_list
        p1, p2, p3 = [p.detach() for p in proto_list]

        tok1 = torch.cat([p1, p2, p3], dim=0)
        tok2 = torch.cat([p2, p3], dim=0)
        tok3 = p3

        return [
            self[0](f1, tok1),
            self[1](f2, tok2),
            self[2](f3, tok3),
        ]


class ContextRefineFusion(nn.Module):
    """Residually inject upsampled APE context into image refinement features."""

    def __init__(self, image_channels=32, context_channels=64):
        super().__init__()
        self.context_residual = nn.Sequential(
            nn.Conv3d(image_channels + context_channels, image_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(image_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image_feat, context_feat):
        return image_feat + self.context_residual(torch.cat([image_feat, context_feat], dim=1))


class RawMedSAMGuidance(nn.Module):
    """Frozen MedSAM2 encoder plus trainable 256->64 feature projectors."""

    def __init__(self, sam2_cfg, sam2_ckpt, sam2_repo=None, device='cuda'):
        super().__init__()
        self.encoder = MedSAM2Encoder3D(
            cfg_path=sam2_cfg,
            ckpt_path=sam2_ckpt,
            sam2_repo=sam2_repo,
            device=device,
        )
        self.projector = RawMedSAMProjector()

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def forward(self, volume):
        with torch.no_grad():
            feats = self.encoder(volume)
        return self.projector(feats)


class AnatomyGuidance(nn.Module):
    """Loads a pretrained APE checkpoint and exposes R/A pyramid maps."""

    def __init__(
        self,
        sam2_cfg,
        sam2_ckpt,
        ape_ckpt,
        sam2_repo=None,
        ape_k_list=(24, 10, 6),
        ape_temp=0.3,
        ape_channels=None,
        device='cuda',
    ):
        super().__init__()

        if not os.path.exists(ape_ckpt):
            raise FileNotFoundError(f"APE checkpoint not found: {ape_ckpt}")

        self.encoder = MedSAM2Encoder3D(
            cfg_path=sam2_cfg,
            ckpt_path=sam2_ckpt,
            sam2_repo=sam2_repo,
            device=device,
        )

        checkpoint = torch.load(ape_ckpt, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        inferred_channels = _infer_ape_channels_from_state_dict(state_dict, default=64)
        if ape_channels is None:
            ape_channels = inferred_channels
        elif int(ape_channels) != int(inferred_channels):
            raise ValueError(
                f"APE channel mismatch: --ape-channels={ape_channels}, "
                f"but checkpoint appears to use {inferred_channels} channels."
            )
        self.ape_channels = int(ape_channels)
        self.ape = APEGuidanceHead(k_list=ape_k_list, temp=ape_temp, channels=self.ape_channels)
        self.ape.load_state_dict(state_dict, strict=True)

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.ape.parameters():
            p.requires_grad = False

        self.encoder.eval()
        self.ape.eval()

    @torch.no_grad()
    def forward(self, volume, return_attention=False, return_projected=False):
        feats = self.encoder(volume)
        R_list, A_list, proj_feats = self.ape(feats)
        if return_attention:
            if return_projected:
                return R_list, A_list, proj_feats
            return R_list, A_list
        if return_projected:
            return R_list, proj_feats
        return R_list


def _resize_flow_with_scale(flow, target_size):
    """
    Resize a flow field and scale displacement magnitudes for each spatial axis.
    flow: [B, 3, D, H, W]
    target_size: tuple/list (D, H, W)
    """
    src_d, src_h, src_w = flow.shape[2:]
    tgt_d, tgt_h, tgt_w = target_size
    resized = nnf.interpolate(flow, size=target_size, mode='trilinear', align_corners=True)

    scale_d = float(tgt_d) / float(src_d)
    scale_h = float(tgt_h) / float(src_h)
    scale_w = float(tgt_w) / float(src_w)

    resized[:, 0, ...] = resized[:, 0, ...] * scale_d
    resized[:, 1, ...] = resized[:, 1, ...] * scale_h
    resized[:, 2, ...] = resized[:, 2, ...] * scale_w
    return resized


def _get_or_build_stn(cache, size_dhw, device):
    """
    Create (or reuse) SpatialTransformer by runtime shape.
    size_dhw: tuple/list (D, H, W)
    """
    key = f"{int(size_dhw[0])}_{int(size_dhw[1])}_{int(size_dhw[2])}"
    if key not in cache:
        cache[key] = SpatialTransformer([int(size_dhw[0]), int(size_dhw[1]), int(size_dhw[2])]).to(device)
    return cache[key]


class RPNet(LoadableModel):
    @store_config_args
    def __init__(
        self,
        size=(80, 96, 80),
        in_channel=1,
        first_channel=8,
        use_ape=False,
        ape_ckpt=None,
        sam2_cfg=None,
        sam2_ckpt=None,
        sam2_repo=None,
        ape_k_list=(24, 10, 6),
        ape_temp=0.3,
        ape_channels=None,
        use_raw_medsam=False,
        ape_use_raw_fusion=False,
        ape_use_gated_raw_fusion=False,
        ape_use_eadf=False,
        ape_use_half_refine=False,
        ape_use_context_refine=False,
        ape_use_proto_attn=False,
        raw_medsam_use_refine=False,
    ):
        super(RPNet, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder4 = DecoderBlock(x_channel=c * 4, y_channel=c * 4, out_channel=c * 4)
        self.decoder3 = DecoderBlock(x_channel=c * 4, y_channel=c * 4, out_channel=c * 4)
        self.decoder2 = DecoderBlock(x_channel=c * 2, y_channel=c * 2, out_channel=c * 4)
        self.decoder1 = DecoderBlock(x_channel=c, y_channel=c, out_channel=c * 2)
        self.size = size
        self.use_ape = bool(use_ape)
        self.use_raw_medsam = bool(use_raw_medsam)
        self.ape_use_raw_fusion = bool(ape_use_raw_fusion)
        self.ape_use_gated_raw_fusion = bool(ape_use_gated_raw_fusion)
        self.ape_use_eadf = bool(ape_use_eadf)
        self.ape_use_half_refine = bool(ape_use_half_refine)
        self.ape_use_context_refine = bool(ape_use_context_refine)
        self.ape_use_proto_attn = bool(ape_use_proto_attn)
        self.raw_medsam_use_refine = bool(raw_medsam_use_refine)
        self.ape_channels = ape_channels
        if self.use_ape and self.use_raw_medsam:
            raise ValueError("use_ape and use_raw_medsam are mutually exclusive.")
        fusion_modes = [self.ape_use_raw_fusion, self.ape_use_gated_raw_fusion, self.ape_use_eadf]
        if sum(int(mode) for mode in fusion_modes) > 1:
            raise ValueError("ape_use_raw_fusion, ape_use_gated_raw_fusion, and ape_use_eadf are mutually exclusive.")

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        if self.use_ape:
            if ape_ckpt is None or sam2_cfg is None or sam2_ckpt is None:
                raise ValueError(
                    "use_ape=True requires ape_ckpt, sam2_cfg, and sam2_ckpt."
                )
            guide_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.anatomy_guide = AnatomyGuidance(
                sam2_cfg=sam2_cfg,
                sam2_ckpt=sam2_ckpt,
                ape_ckpt=ape_ckpt,
                sam2_repo=sam2_repo,
                ape_k_list=ape_k_list,
                ape_temp=ape_temp,
                ape_channels=ape_channels,
                device=guide_device,
            )
            self.ape_channels = self.anatomy_guide.ape_channels
            # 3-level APE pyramid -> flow residual decoders
            self.ape_decoder3 = DecoderBlock(x_channel=self.ape_channels, y_channel=self.ape_channels, out_channel=self.ape_channels)
            self.ape_decoder2 = DecoderBlock(x_channel=self.ape_channels, y_channel=self.ape_channels, out_channel=self.ape_channels)
            self.ape_decoder1 = DecoderBlock(x_channel=self.ape_channels, y_channel=self.ape_channels, out_channel=self.ape_channels)
            if self.ape_use_raw_fusion:
                self.ape_raw_fusion = PyramidFeatureFusion(channels=self.ape_channels)
            if self.ape_use_gated_raw_fusion:
                self.ape_gated_raw_fusion = PyramidGatedResidualFusion(channels=self.ape_channels)
            if self.ape_use_eadf:
                self.ape_eadf = EntropyAwareDetailFusion(channels=self.ape_channels)
            if self.ape_use_proto_attn:
                self.proto_attn = AnatomicalPrototypeAttention(channels=self.ape_channels, num_heads=4)
            if self.ape_use_half_refine:
                self.half_refine_encoder = FineRefineEncoder(in_channel=in_channel, out_channel=32)
                self.half_refine_decoder = DecoderBlock(x_channel=32, y_channel=32, out_channel=32)
            self.fine_refine_encoder = FineRefineEncoder(in_channel=in_channel, out_channel=32)
            if self.ape_use_context_refine:
                self.context_refine_fusion = ContextRefineFusion(image_channels=32, context_channels=self.ape_channels)
            self.fine_refine_decoder = DecoderBlock(x_channel=32, y_channel=32, out_channel=32)
            self.ape_stn_cache = nn.ModuleDict()
        elif self.use_raw_medsam:
            if sam2_cfg is None or sam2_ckpt is None:
                raise ValueError(
                    "use_raw_medsam=True requires sam2_cfg and sam2_ckpt."
                )
            guide_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.raw_medsam_guide = RawMedSAMGuidance(
                sam2_cfg=sam2_cfg,
                sam2_ckpt=sam2_ckpt,
                sam2_repo=sam2_repo,
                device=guide_device,
            )
            self.raw_decoder3 = DecoderBlock(x_channel=64, y_channel=64, out_channel=64)
            self.raw_decoder2 = DecoderBlock(x_channel=64, y_channel=64, out_channel=64)
            self.raw_decoder1 = DecoderBlock(x_channel=64, y_channel=64, out_channel=64)
            if self.raw_medsam_use_refine:
                self.fine_refine_encoder = FineRefineEncoder(in_channel=in_channel, out_channel=32)
                self.fine_refine_decoder = DecoderBlock(x_channel=32, y_channel=32, out_channel=32)
            self.raw_stn_cache = nn.ModuleDict()
        else:
            self.anatomy_guide = None
            self.raw_medsam_guide = None

    def forward(self, x, y):
        if self.use_ape:
            with torch.no_grad():
                self.anatomy_guide.eval()
                need_proj = self.ape_use_raw_fusion or self.ape_use_gated_raw_fusion or self.ape_use_eadf
                if need_proj:
                    if self.ape_use_eadf:
                        rx_list, ax_list, px_list = self.anatomy_guide(x, return_attention=True, return_projected=True)
                        ry_list, ay_list, py_list = self.anatomy_guide(y, return_attention=True, return_projected=True)
                    else:
                        rx_list, px_list = self.anatomy_guide(x, return_projected=True)
                        ry_list, py_list = self.anatomy_guide(y, return_projected=True)
                else:
                    rx_list = self.anatomy_guide(x)
                    ry_list = self.anatomy_guide(y)

            if self.ape_use_raw_fusion:
                rx_list = self.ape_raw_fusion(rx_list, px_list)
                ry_list = self.ape_raw_fusion(ry_list, py_list)
            elif self.ape_use_gated_raw_fusion:
                rx_list = self.ape_gated_raw_fusion(rx_list, px_list)
                ry_list = self.ape_gated_raw_fusion(ry_list, py_list)
            elif self.ape_use_eadf:
                rx_list = self.ape_eadf(rx_list, ax_list, px_list)
                ry_list = self.ape_eadf(ry_list, ay_list, py_list)

            # Use APE runtime output order directly: [B, C, D, H, W].
            rx1, rx2, rx3 = rx_list[0], rx_list[1], rx_list[2]
            ry1, ry2, ry3 = ry_list[0], ry_list[1], ry_list[2]

            if self.ape_use_proto_attn:
                rx1, rx2, rx3 = self.proto_attn(
                    [rx1, rx2, rx3],
                    self.anatomy_guide.ape.protos,
                )
                ry1, ry2, ry3 = self.proto_attn(
                    [ry1, ry2, ry3],
                    self.anatomy_guide.ape.protos,
                )

            stn3 = _get_or_build_stn(self.ape_stn_cache, rx3.shape[2:], rx3.device)
            stn2 = _get_or_build_stn(self.ape_stn_cache, rx2.shape[2:], rx2.device)
            stn1 = _get_or_build_stn(self.ape_stn_cache, rx1.shape[2:], rx1.device)

            # 3-level coarse-to-fine registration on APE pyramid
            flow3 = self.ape_decoder3(rx3, ry3)
            w_rx3 = stn3(rx3, flow3)
            flow3 = self.ape_decoder3(w_rx3, ry3) + flow3

            flow2_up = _resize_flow_with_scale(flow3, rx2.shape[2:])
            w_rx2 = stn2(rx2, flow2_up)
            dflow2 = self.ape_decoder2(w_rx2, ry2)
            flow2 = stn2(flow2_up, dflow2) + dflow2

            flow1_up = _resize_flow_with_scale(flow2, rx1.shape[2:])
            w_rx1 = stn1(rx1, flow1_up)
            dflow1 = self.ape_decoder1(w_rx1, ry1)
            flow1 = stn1(flow1_up, dflow1) + dflow1

            if self.ape_use_half_refine:
                half_size = (
                    int(x.shape[2]),
                    max(1, int(x.shape[3]) // 2),
                    max(1, int(x.shape[4]) // 2),
                )
                flow_half_up = _resize_flow_with_scale(flow1, half_size)
                x_half = nnf.interpolate(x, size=half_size, mode='trilinear', align_corners=True)
                y_half = nnf.interpolate(y, size=half_size, mode='trilinear', align_corners=True)
                fx_half = self.half_refine_encoder(x_half)
                fy_half = self.half_refine_encoder(y_half)
                stn_half = _get_or_build_stn(self.ape_stn_cache, half_size, x.device)
                w_fx_half = stn_half(fx_half, flow_half_up)
                dflow_half = self.half_refine_decoder(w_fx_half, fy_half)
                flow1 = stn_half(flow_half_up, dflow_half) + dflow_half

            # Fine refinement at image resolution with 32-channel features.
            refine_size = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
            flow_ref_up = _resize_flow_with_scale(flow1, refine_size)
            fx_ref = self.fine_refine_encoder(x)
            fy_ref = self.fine_refine_encoder(y)
            if self.ape_use_context_refine:
                rx1_ref = nnf.interpolate(rx1, size=refine_size, mode='trilinear', align_corners=True)
                ry1_ref = nnf.interpolate(ry1, size=refine_size, mode='trilinear', align_corners=True)
                fx_ref = self.context_refine_fusion(fx_ref, rx1_ref)
                fy_ref = self.context_refine_fusion(fy_ref, ry1_ref)
            stn_ref = _get_or_build_stn(self.ape_stn_cache, refine_size, x.device)
            w_fx_ref = stn_ref(fx_ref, flow_ref_up)
            dflow_ref = self.fine_refine_decoder(w_fx_ref, fy_ref)
            flow_ref = stn_ref(flow_ref_up, dflow_ref) + dflow_ref

            # Upsample refined flow to image resolution.
            flowall = _resize_flow_with_scale(flow_ref, x.shape[2:])
            stn_img = _get_or_build_stn(self.ape_stn_cache, x.shape[2:], x.device)
            warped_x = stn_img(x, flowall)
            return warped_x, flowall

        if self.use_raw_medsam:
            rx_list = self.raw_medsam_guide(x)
            ry_list = self.raw_medsam_guide(y)

            rx1, rx2, rx3 = rx_list[0], rx_list[1], rx_list[2]
            ry1, ry2, ry3 = ry_list[0], ry_list[1], ry_list[2]

            stn3 = _get_or_build_stn(self.raw_stn_cache, rx3.shape[2:], rx3.device)
            stn2 = _get_or_build_stn(self.raw_stn_cache, rx2.shape[2:], rx2.device)
            stn1 = _get_or_build_stn(self.raw_stn_cache, rx1.shape[2:], rx1.device)

            flow3 = self.raw_decoder3(rx3, ry3)
            w_rx3 = stn3(rx3, flow3)
            flow3 = self.raw_decoder3(w_rx3, ry3) + flow3

            flow2_up = _resize_flow_with_scale(flow3, rx2.shape[2:])
            w_rx2 = stn2(rx2, flow2_up)
            dflow2 = self.raw_decoder2(w_rx2, ry2)
            flow2 = stn2(flow2_up, dflow2) + dflow2

            flow1_up = _resize_flow_with_scale(flow2, rx1.shape[2:])
            w_rx1 = stn1(rx1, flow1_up)
            dflow1 = self.raw_decoder1(w_rx1, ry1)
            flow1 = stn1(flow1_up, dflow1) + dflow1

            if self.raw_medsam_use_refine:
                refine_size = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
                flow_ref_up = _resize_flow_with_scale(flow1, refine_size)
                fx_ref = self.fine_refine_encoder(x)
                fy_ref = self.fine_refine_encoder(y)
                stn_ref = _get_or_build_stn(self.raw_stn_cache, refine_size, x.device)
                w_fx_ref = stn_ref(fx_ref, flow_ref_up)
                dflow_ref = self.fine_refine_decoder(w_fx_ref, fy_ref)
                flow1 = stn_ref(flow_ref_up, dflow_ref) + dflow_ref

            flowall = _resize_flow_with_scale(flow1, x.shape[2:])
            stn_img = _get_or_build_stn(self.raw_stn_cache, x.shape[2:], x.device)
            warped_x = stn_img(x, flowall)
            return warped_x, flowall

        fx1, fx2, fx3, fx4 = self.encoder(x)
        fy1, fy2, fy3, fy4 = self.encoder(y)

        ar = 1
        br = 1
        cr = 1
        dr = 1

        wx4 = fx4

        flowall =  None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow)+flow

        flowall = self.up(2*flowall)
        for bb in range(br):
            wx3 = self.transformer[2](fx3, flowall)
            flow = self.decoder3(wx3, fy3)
            flowall = self.transformer[2](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for cc in range(cr):
            wx2 = self.transformer[1](fx2, flowall)
            flow = self.decoder2(wx2, fy2)
            flowall = self.transformer[1](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for dd in range(dr):
            wx1 = self.transformer[0](fx1, flowall)
            flow = self.decoder1(wx1, fy1)
            flowall = self.transformer[0](flowall, flow) + flow
        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall

class IIRPNet(LoadableModel):
    @store_config_args
    def __init__(
        self,
        size=(80, 96, 80),
        in_channel=1,
        first_channel=8,
        use_ape=False,
        ape_ckpt=None,
        sam2_cfg=None,
        sam2_ckpt=None,
        sam2_repo=None,
        ape_k_list=(24, 10, 6),
        ape_temp=0.3,
        ape_channels=None,
    ):
        super(IIRPNet, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder4 = DecoderBlock(x_channel=c * 4, y_channel=c * 4, out_channel=c * 4)
        self.decoder3 = DecoderBlock(x_channel=c * 4, y_channel=c * 4, out_channel=c * 4)
        self.decoder2 = DecoderBlock(x_channel=c * 2, y_channel=c * 2, out_channel=c * 4)
        self.decoder1 = DecoderBlock(x_channel=c, y_channel=c, out_channel=c * 2)
        self.size = size
        self.use_ape = bool(use_ape)

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        if self.use_ape:
            if ape_ckpt is None or sam2_cfg is None or sam2_ckpt is None:
                raise ValueError(
                    "use_ape=True requires ape_ckpt, sam2_cfg, and sam2_ckpt."
                )
            guide_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.anatomy_guide = AnatomyGuidance(
                sam2_cfg=sam2_cfg,
                sam2_ckpt=sam2_ckpt,
                ape_ckpt=ape_ckpt,
                sam2_repo=sam2_repo,
                ape_k_list=ape_k_list,
                ape_temp=ape_temp,
                ape_channels=ape_channels,
                device=guide_device,
            )
            self.ape_channels = self.anatomy_guide.ape_channels
            self.ape_decoder3 = DecoderBlock(x_channel=self.ape_channels, y_channel=self.ape_channels, out_channel=self.ape_channels)
            self.ape_decoder2 = DecoderBlock(x_channel=self.ape_channels, y_channel=self.ape_channels, out_channel=self.ape_channels)
            self.ape_decoder1 = DecoderBlock(x_channel=self.ape_channels, y_channel=self.ape_channels, out_channel=self.ape_channels)
            # Add one fine refinement stage at half spatial resolution with 32 channels.
            self.fine_refine_encoder = FineRefineEncoder(in_channel=in_channel, out_channel=32)
            self.fine_refine_decoder = DecoderBlock(x_channel=32, y_channel=32, out_channel=32)
            self.ape_stn_cache = nn.ModuleDict()
        else:
            self.anatomy_guide = None

    def normalized_cross_correlation(self, img1, img2):
        # 计算均值
        mean_img1 = torch.mean(img1)
        mean_img2 = torch.mean(img2)

        # 计算标准差
        std_img1 = torch.std(img1)
        std_img2 = torch.std(img2)

        # 计算NCC
        ncc = torch.mean((img1 - mean_img1) * (img2 - mean_img2) / (std_img1 * std_img2))

        return ncc

    def pnsr(self, original_image, processed_image):
        # 确保输入张量的数据类型为浮点数
        original_image = original_image.float()
        processed_image = processed_image.float()

        # 计算均方误差（MSE）
        mse = torch.mean((original_image - processed_image) ** 2)

        # 如果MSE为0，PSNR无穷大，这里我们使用一个很小的非零值替代0以避免错误
        if mse == 0:
            return float('inf')

        # 计算PSNR
        max_intensity = 1.0  # 假设图像像素值的范围在0到255之间
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)

        return psnr

    def forward(self, x, y):
        if self.use_ape:
            with torch.no_grad():
                self.anatomy_guide.eval()
                rx_list = self.anatomy_guide(x)
                ry_list = self.anatomy_guide(y)

            rx1, rx2, rx3 = rx_list[0], rx_list[1], rx_list[2]
            ry1, ry2, ry3 = ry_list[0], ry_list[1], ry_list[2]

            stn3 = _get_or_build_stn(self.ape_stn_cache, rx3.shape[2:], rx3.device)
            stn2 = _get_or_build_stn(self.ape_stn_cache, rx2.shape[2:], rx2.device)
            stn1 = _get_or_build_stn(self.ape_stn_cache, rx1.shape[2:], rx1.device)

            current_iter = []

            flow3 = self.ape_decoder3(rx3, ry3)
            w_rx3 = stn3(rx3, flow3)
            flow3 = self.ape_decoder3(w_rx3, ry3) + flow3
            current_iter.append(2)

            flow2_up = _resize_flow_with_scale(flow3, rx2.shape[2:])
            w_rx2 = stn2(rx2, flow2_up)
            dflow2 = self.ape_decoder2(w_rx2, ry2)
            flow2 = stn2(flow2_up, dflow2) + dflow2
            current_iter.append(1)

            flow1_up = _resize_flow_with_scale(flow2, rx1.shape[2:])
            w_rx1 = stn1(rx1, flow1_up)
            dflow1 = self.ape_decoder1(w_rx1, ry1)
            flow1 = stn1(flow1_up, dflow1) + dflow1
            current_iter.append(1)

            # Fine refinement at image resolution (e.g., [192, 112, 80]) with 32-channel features.
            refine_size = (int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
            flow_ref_up = _resize_flow_with_scale(flow1, refine_size)
            fx_ref = self.fine_refine_encoder(x)
            fy_ref = self.fine_refine_encoder(y)
            stn_ref = _get_or_build_stn(self.ape_stn_cache, refine_size, x.device)
            w_fx_ref = stn_ref(fx_ref, flow_ref_up)
            dflow_ref = self.fine_refine_decoder(w_fx_ref, fy_ref)
            flow_ref = stn_ref(flow_ref_up, dflow_ref) + dflow_ref
            current_iter.append(1)

            flowall = _resize_flow_with_scale(flow_ref, x.shape[2:])
            stn_img = _get_or_build_stn(self.ape_stn_cache, x.shape[2:], x.device)
            warped_x = stn_img(x, flowall)
            return warped_x, flowall, current_iter

        fx1, fx2, fx3, fx4 = self.encoder(x)
        fy1, fy2, fy3, fy4 = self.encoder(y)

        ar = 10
        br = 10
        cr = 10
        dr = 10

        pa = pb = pc = pd = 0

        current_iter = []
        wx4 = fx4
        mse_a = 100 #mse mae
        ncc_a = 0 #ncc,pnsr
        delta1 =0.005
        delta2 =0.005
        delta3 =0.005
        delta4 =0.005

        flowall =  None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            previous_flow = flowall
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow)+flow
            wx4 = self.transformer[3](fx4, flowall)
            # not use in train
            flowx4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)(8*flowall)
            mx4 = self.transformer[0](x, flowx4)
            ncc = self.normalized_cross_correlation(mx4, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pa += 1
        current_iter.append(pa)

        flowall = self.up(2*flowall)
        mse_a = 100
        ncc_a = 0
        previous_flow = flowall
        for bb in range(br):
            previous_flow = flowall
            wx3 = self.transformer[2](fx3, flowall)
            # not use in train
            flowx3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)(4*flowall)
            mx3 = self.transformer[0](x, flowx3)
            ncc = self.normalized_cross_correlation(mx3, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pb += 1

            flow = self.decoder3(wx3, fy3)
            previous_flow = flowall
            flowall = self.transformer[2](flowall, flow) + flow
        current_iter.append(pb)

        flowall = self.up(2 * flowall)
        previous_flow = flowall
        mse_a = 100
        ncc_a = 0
        for cc in range(cr):
            previous_flow = flowall
            wx2 = self.transformer[1](fx2, flowall)
            # not use in train
            flowx2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(2*flowall)
            mx2 = self.transformer[0](x, flowx2)
            ncc = self.normalized_cross_correlation(mx2, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pc += 1

            flow = self.decoder2(wx2, fy2)
            previous_flow = flowall
            flowall = self.transformer[1](flowall, flow) + flow
        current_iter.append(pc)

        flowall = self.up(2 * flowall)
        previous_flow = flowall
        mse_a = 100
        ncc_a = 0
        for dd in range(dr):
            previous_flow = flowall
            wx1 = self.transformer[0](fx1, flowall)
            # not use in train
            mx = self.transformer[0](x, flowall)
            ncc = self.normalized_cross_correlation(mx, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pd += 1

            flow = self.decoder1(wx1, fy1)
            previous_flow = flowall
            flowall = self.transformer[0](flowall, flow) + flow
        current_iter.append(pd)

        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall, current_iter
