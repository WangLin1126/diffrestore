"""DDPM/NCSN-style U-Net denoiser (torch-only), used as the non-hot x0-predictor F_theta.

Copied and lightly cleaned from the Cold-Diffusion / 'Diffusion Models Beat GANs' U-Net
(only torch + math deps). Predicts a clean image x0 from a degraded state x_t and level t.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim):
    half = dim // 2
    emb = math.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def swish(x):
    return x * torch.sigmoid(x)


def Normalize(c):
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 2, 0)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, temb_c, dropout):
        super().__init__()
        self.in_c, self.out_c = in_c, out_c
        self.norm1 = Normalize(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.temb_proj = nn.Linear(temb_c, out_c)
        self.norm2 = Normalize(out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        if in_c != out_c:
            self.nin_shortcut = nn.Conv2d(in_c, out_c, 1, 1, 0)

    def forward(self, x, temb):
        h = self.conv1(swish(self.norm1(x)))
        h = h + self.temb_proj(swish(temb))[:, :, None, None]
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        if self.in_c != self.out_c:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = Normalize(c)
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.proj_out = nn.Conv2d(c, c, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, w = q.shape
        q = q.reshape(b, c, hh * w).permute(0, 2, 1)
        k = k.reshape(b, c, hh * w)
        w_ = torch.bmm(q, k) * (c ** -0.5)
        w_ = torch.softmax(w_, dim=2).permute(0, 2, 1)
        v = v.reshape(b, c, hh * w)
        h = torch.bmm(v, w_).reshape(b, c, hh, w)
        return x + self.proj_out(h)


class UNet(nn.Module):
    def __init__(self, ch=64, out_ch=3, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
                 attn_resolutions=(16,), dropout=0.0, in_channels=3, resolution=128):
        super().__init__()
        self.ch = ch
        self.temb_ch = ch * 4
        self.num_res = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([nn.Linear(ch, self.temb_ch),
                                         nn.Linear(self.temb_ch, self.temb_ch)])
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        curr = resolution
        in_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i in range(self.num_res):
            block, attn = nn.ModuleList(), nn.ModuleList()
            bin_, bout = ch * in_mult[i], ch * ch_mult[i]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(bin_, bout, self.temb_ch, dropout))
                bin_ = bout
                if curr in attn_resolutions:
                    attn.append(AttnBlock(bin_))
            d = nn.Module(); d.block = block; d.attn = attn
            if i != self.num_res - 1:
                d.downsample = Downsample(bin_); curr //= 2
            self.down.append(d)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(bin_, bin_, self.temb_ch, dropout)
        self.mid.attn_1 = AttnBlock(bin_)
        self.mid.block_2 = ResnetBlock(bin_, bin_, self.temb_ch, dropout)

        self.up = nn.ModuleList()
        for i in reversed(range(self.num_res)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            bout = ch * ch_mult[i]
            for j in range(num_res_blocks + 1):
                skip = ch * (in_mult[i] if j == num_res_blocks else ch_mult[i])
                block.append(ResnetBlock(bin_ + skip, bout, self.temb_ch, dropout))
                bin_ = bout
                if curr in attn_resolutions:
                    attn.append(AttnBlock(bin_))
            u = nn.Module(); u.block = block; u.attn = attn
            if i != 0:
                u.upsample = Upsample(bin_); curr *= 2
            self.up.insert(0, u)

        self.norm_out = Normalize(bin_)
        self.conv_out = nn.Conv2d(bin_, out_ch, 3, 1, 1)

    def forward(self, x, t):
        temb = self.temb.dense[0](timestep_embedding(t, self.ch))
        temb = self.temb.dense[1](swish(temb))
        hs = [self.conv_in(x)]
        for i in range(self.num_res):
            for j in range(self.num_res_blocks):
                h = self.down[i].block[j](hs[-1], temb)
                if len(self.down[i].attn) > 0:
                    h = self.down[i].attn[j](h)
                hs.append(h)
            if i != self.num_res - 1:
                hs.append(self.down[i].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h, temb)), temb)
        for i in reversed(range(self.num_res)):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].block[j](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i].attn) > 0:
                    h = self.up[i].attn[j](h)
            if i != 0:
                h = self.up[i].upsample(h)
        return self.conv_out(swish(self.norm_out(h)))
