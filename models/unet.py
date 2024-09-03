import math
import torch
import torch.nn as nn
import torch.nn.functional

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm
# 新加的
from dense_layer import dense
from layers import ddpm_conv1x1 as conv1x1
from layers import ddpm_conv3x3 as conv3x3
from layers import default_init
from layers import get_timestep_embedding
from DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
import numpy as np
import up_or_down_sampling
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):              # 只改变x的h和w（减半）

        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class Res_DWTBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.resblock = ResnetBlock(in_channels=in_channels,
                                    out_channels=in_channels,
                                    temb_channels=temb_channels,
                                    dropout=dropout)
        self.dwt = DWT_2D("haar")

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv3 = torch.nn.Conv2d(in_channels * 4,  # 修改这里，确保通道数匹配
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv3_1 = torch.nn.Conv2d(in_channels,  # 修改这里，确保通道数匹配
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.upChannel = torch.nn.Conv2d(in_channels *3,  # 修改这里，确保通道数匹配
                                         (out_channels + 64)*3,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

    def forward(self, x, temb):
        fin = x           # torch.Size([8, 64, 64, 64])
        # conv + dwt
        # 1：in=64,out=64, 2.in=64,out=128,3.in=128,out=192,4.in=192,out=256

        if fin.size()[-1] != 1:

            x1 = torch.cat(self.dwt(fin), dim=1) / 2.               # torch.Size([8, 256, 32, 32])

            x1 = self.conv3(x1)

        else:
            x1 = self.conv3_1(fin)
        # ch_mult: [1, 2, 3, 4]

        # res
        x2 = self.resblock(x, temb)
        x2 = self.conv1(x2)
        if x2.size()[-1] != 1:
            xLL, xLH, xHL, xHH = self.dwt(x2)        # xLL = torch.Size([8, 64, 32, 32])

        else:
            xLL = x2
            xLH = x2
            xHL = x2
            xHH = x2
        x2_ll = xLL
        x2_hh = torch.cat((xLH, xHL, xHH), dim=1)

        x2_ll = self.resblock(x2_ll, temb)
        x2_ll = self.conv2(x2_ll)

        # 把他们的h和w都翻倍，不然h和w在每个DWT+DOWN里都会减半h和w
        # x1 = torch.nn.functional.interpolate(x1, scale_factor=2.0, mode="nearest")
        # x2_ll = torch.nn.functional.interpolate(x2_ll, scale_factor=2.0, mode="nearest")
        # x2_hh = torch.nn.functional.interpolate(x2_hh, scale_factor=2.0, mode="nearest")

        x2_hh = self.upChannel(x2_hh)

        return x1 + x2_ll, x2_hh


class Res_IWTBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.resblock = ResnetBlock(in_channels=in_channels,
                                    out_channels=in_channels,
                                    temb_channels=temb_channels,
                                    dropout=dropout)
        self.iwt = IDWT_2D("haar")

        self.conv1 = torch.nn.Conv2d(in_channels,
                                     in_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels,  # 修改这里，确保通道数匹配
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.conv3 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv3_1 = torch.nn.Conv2d(256,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.sub_hw = torch.nn.Conv2d(out_channels,  # 修改这里，确保通道数匹配
                                       out_channels,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1)

    def forward(self, x, temb, high):
        ll = x

        # parts = high.chunk(3, dim=1)
        # hlh, hhl, hhh = parts[0], parts[1], parts[2]
        high_ch = high.size()[1]
        hlh = high[:, :high_ch//3, :, :]
        hhl = high[:, high_ch//3:2*high_ch//3, :, :]
        hhh = high[:, 2*high_ch//3:, :, :]

        # hlh = high[:, :3, :, :]
        # hhl = high[:, 3:6, :, :]
        # hhh = high[:, 6:, :, :]

        # iwt + conv3

        x1 = self.iwt(ll, hlh, hhl, hhh)

        if x1.size()[1] == 256:
            x1 = self.conv3_1(x1)
        else:
            x1 = self.conv3(x1)
        # Res
        x2 = self.resblock(ll, temb)
        x2 = self.conv1(x2)
        x2 = self.iwt(x2, hlh, hhl, hhh)
        x2 = self.resblock(x2, temb)
        x2 = self.conv2(x2)

        # x_res = self.sub_hw(x1+x2)

        return x1 + x2


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level == 2:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                # down.downsample = Downsample(block_in, resamp_with_conv)
                down.downsample = Res_DWTBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                # print("上采样Res的输入输出通道数")
                # print(block_in+skip_in)
                # print(block_out)
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level == 2:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # up.upsample = Upsample(block_in, resamp_with_conv)
                # print("上采样模块中的RESIWT的输入输出通道数")
                # print(block_in)
                up.upsample = Res_IWTBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout)


            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        # assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        skip_info = []               # 存储新下采样模块的高频信息，作为跳跃连接
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                fea, skip = self.down[i_level].downsample(hs[-1], temb)
                hs.append(fea)
                skip_info.append(skip)

        # middle

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        count = 0
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                # print(f"第{count}个上采样输入尺寸")
                # print(h.size())
                # h = self.up[i_level].upsample(h)
                skip_t = skip_info.pop()
                # print(skip_t.size())
                h = self.up[i_level].upsample(h, temb, skip_t)
                # print(f"第{count}个上采样输出尺寸")
                # print(h.size())
                count = count + 1

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
