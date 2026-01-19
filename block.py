from timm.models.layers import trunc_normal_, DropPath
import numpy as np
from norms import *
from ecb import SeqConv3x3
from einops import rearrange
import torch.nn.functional as F
import numbers
import math
from mmcv.cnn import build_norm_layer
from trans_form import dim3_4, dim4_3


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, kernel_size=3, padding=1, stride=1, dilation=1):
            super().__init__()

            self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=dim_in)
            self.norm_layer = nn.GroupNorm(4, dim_in)
            self.conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features * 2, bias=bias),
                                    LayerNorm(hidden_features * 2, 'WithBias'),
                                    nn.ReLU(inplace=True),
        )

        self.spatial_gating_unit = StripBlock(dim)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
        )
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', dim, dim, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', dim, dim, -1)

    def forward(self, x):
        x = dim3_4(x)
        spatial = x
        x = self.project_in(x)

        y =  self.conv1x1_sbx(spatial) + self.conv1x1_sby(spatial)
        y = self.conv(y)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)
        x = F.gelu(x1)*x2
        x = self.project_out(x)
        return dim4_3(x)


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True, patch_size=8, clip_limit=1.0, n_bins=9):
        super(PoolingAttention, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Sequential(
            nn.Conv2d(dim * 5, dim * 5, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim * 5),
            LayerNorm(dim * 5, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 5, dim * 5, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim * 5)
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.bin_proj = nn.Conv2d(n_bins, dim // 2, kernel_size=1, bias=bias)
        self.patch_size = patch_size
        self.n_bins = n_bins
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
        )

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        *_, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def split_into_patches(self, x):
        b, c, h, w = x.shape
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) c (p1 p2)', p1=self.patch_size, p2=self.patch_size)
        n_h, n_w = (h + pad_h) // self.patch_size, (w + pad_w) // self.patch_size
        return patches, (b, c, h, w, pad_h, pad_w, n_h, n_w)

    def merge_patches(self, patches, shape_info):
        b, c, h, w, pad_h, pad_w, n_h, n_w = shape_info
        patches = rearrange(patches, 'b (h w) c (p1 p2) -> b c (h p1) (w p2)', h=n_h, w=n_w, p1=self.patch_size,
                            p2=self.patch_size)
        if pad_h > 0 or pad_w > 0:
            patches = patches[:, :, :h, :w]
        return patches

    def apply_hog_to_patch(self, x_half):
        b, c, h, w = x_half.shape
        gx = F.adaptive_avg_pool2d(x_half,(round(h), round(w)))
        gy = F.adaptive_max_pool2d(x_half,(round(h), round(w)))
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        orientation = torch.atan2(gy, gx)  # [-pi, pi]
        orientation_bin = ((orientation + torch.pi) / (2 * torch.pi) * self.n_bins).long() % self.n_bins
        patches_x, shape_info = self.split_into_patches(x_half)
        patches_mag, _ = self.split_into_patches(magnitude)
        patches_ori, _ = self.split_into_patches(orientation_bin.float())
        b, n_patches, c, patch_pixels = patches_x.shape
        sort_values = torch.zeros_like(patches_x)
        hog_features = torch.zeros(b, n_patches, self.n_bins, device=x_half.device)
        for i in range(self.n_bins):
            bin_mask = (patches_ori == i).float()
            bin_magnitude = patches_mag * bin_mask
            sort_values += bin_magnitude * (i + 1)
            hog_features[..., i] = bin_magnitude.mean(dim=[-1, -2])

        hog_features = hog_features / (hog_features.sum(dim=-1, keepdim=True) + 1e-8)
        _, sort_indices = sort_values.sum(dim=2, keepdim=True).expand_as(patches_x).sort(dim=-1)
        patches_x_sorted = torch.gather(patches_x, -1, sort_indices)
        x_half_processed = self.merge_patches(patches_x_sorted, shape_info)
        return x_half_processed, sort_indices, hog_features, shape_info

    def forward(self, x):
        b, n, c = x.shape
        h = w =int(n**0.5)
        x = x.permute(0,2,1).reshape(b,c,h,w)
        half_c = c // 2
        x = self.qkv_dwconv(self.qkv(x))
        x_half = x[:, :half_c]
        x_half_processed, idx_patch, hog_features, shape_info = self.apply_hog_to_patch(x_half)
        b, n_patches, n_bins = hog_features.shape
        n_h = shape_info[-2]  # int(math.sqrt(n_patches))
        n_w = shape_info[-1]
        hog_map = rearrange(hog_features, 'b (nh nw) bins -> b bins nh nw', nh=n_h, nw=n_w).contiguous()
        hog_map = self.bin_proj(hog_map)
        hog_map = F.interpolate(hog_map, size=(h, w), mode='bilinear')
        x = torch.cat((x_half_processed + hog_map, x[:, half_c:]), dim=1)

        q1, k1, q2, k2, v = x.chunk(5, dim=1)  # b,c,x,x
        gx = self.conv5(v)
        gy = F.adaptive_max_pool2d(v,(round(h), round(w)))
        v = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).view(b, c, -1)
        orientation = torch.atan2(gy, gx).view(b, c, -1)  # [-pi, pi]

        orientation_norm = ((orientation + torch.pi) / (2 * torch.pi))
        weighted_magnitude = v * orientation_norm
        _, idx = weighted_magnitude.sum(dim=1).sort(dim=-1)
        idx = idx.unsqueeze(1).expand(b, c, -1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)

        out_replace = out[:, :half_c]
        patches_out, shape_info = self.split_into_patches(out_replace)
        patches_out = torch.scatter(patches_out, -1, idx_patch, patches_out)
        out_replace = self.merge_patches(patches_out, shape_info)
        out[:, :half_c] = out_replace
        return out.reshape(b,c,-1).permute(0,2,1)

class Block(nn.Module):

    def __init__(self, dim, num_heads=8,ffn_expansion_factor=2, qkv_bias=False,drop_path=0., norm_layer=nn.LayerNorm, layerscale_value=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(dim, num_heads, bias = qkv_bias, ifBox=True, patch_size=8, clip_limit=1.0, n_bins=9)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias=qkv_bias)
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        res = x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        x = x + (res * self.gamma)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 64, 64).cuda()
    m = Block(64).cuda()
    o = m(x)
    print(o.shape)