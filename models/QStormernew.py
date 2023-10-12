import torch
from torch import nn, einsum
from torchvision import transforms
from torch.nn import functional as F

from torchsummary import summary

from einops import rearrange

from PIL import Image

import matplotlib.pyplot as plt


class MDTA(nn.Module):
    def __init__(self, channels, num_heads, return_qk=False, takes_qk=False):
        super(MDTA, self).__init__()
        
        # informative attributes
        self.takes_qk = takes_qk

        # functional attributes
        self.num_heads = num_heads
        self.return_qk = return_qk
        
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, q_in=None, k_in=None):
        b, c, h, w = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)


        if q_in!=None and k_in!=None:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            q_new = q + q_in
            k_new = k + k_in
            q_new, k_new = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            attn = torch.softmax(torch.matmul(q_new, k_new.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        else:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        if self.return_qk:
            return out, q, k
        else:
            return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor, return_qk=False, takes_qk=False):
        super(TransformerBlock, self).__init__()
        
        # informative attributes
        self.takes_qk = takes_qk

        # functional attributes
        self.return_qk = return_qk

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads, return_qk=return_qk, takes_qk=takes_qk)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, q_in=None, k_in=None):
        b, c, h, w = x.shape
        if q_in == None and k_in == None and self.return_qk:
            att, q, k = self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w))
            x = att + x
        elif q_in == None and k_in == None and not self.return_qk:
            x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w))
        elif q_in != None and k_in != None and self.return_qk:
            att, q, k = self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w), q_in, k_in)
            x = att + x
        else:
            x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w), q_in, k_in)
        
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        
        if self.return_qk:
            return x, q, k
        else:
            return x
        

class TransformerBulk(nn.Module):
    def __init__(self, num_blocks, channels, num_heads, expansion_factor, return_qk=False, takes_qk=False):
        super().__init__()

        # informative attributes
        self.takes_qk = takes_qk

        # functional attributes
        self.return_qk = return_qk
        self.num_blocks = num_blocks

        self.bulk = nn.ModuleList()
        # first block
        self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, False, False))
        self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, True, takes_qk))

        # intermediate blocks
        if num_blocks>2:
            for _ in range(num_blocks-2):
                self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, False, False))
                self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, True, True))
                
        # last block
        self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, False, False))
        self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, return_qk, True))

    def forward(self, x, q_in=None, k_in=None):
        # first block
        x = self.bulk[0](x)
        if q_in==None and k_in==None:
            x, q, k = self.bulk[1](x)
        else:
            x, q, k = self.bulk[1](x, q_in, k_in)

        # intermediate blocks
        if self.num_blocks>2:
            for layer in range(1, self.num_blocks-1):
                x = self.bulk[2 * layer](x)
                x, q, k = self.bulk[2 * layer  + 1](x, q, k)

        # last block
        if self.return_qk:
            x = self.bulk[-2](x)
            x, q, k = self.bulk[-1](x, q, k)
            return x, q, k
        else:
            x = self.bulk[-2](x)
            x = self.bulk[-1](x, q, k)
            return x
        

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.dbody = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.dbody(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.ubody = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.ubody(x)
    

class PASTormer(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_layers=4,
                 expansion_factor=2.66):
        super(PASTormer, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([TransformerBulk(num_tb, num_ch, num_ah, expansion_factor) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])

        self.reduces = nn.ModuleList([nn.Conv2d(2 * channels[i - 1], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(1, len(channels)))])

        self.decoders = nn.ModuleList([TransformerBulk(num_blocks[2], channels[2], num_heads[2], expansion_factor)])
        self.decoders.append(TransformerBulk(num_blocks[1], channels[1], num_heads[1], expansion_factor))
        self.decoders.append(TransformerBulk(num_blocks[0], channels[0], num_heads[0], expansion_factor))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[0], num_heads[0], expansion_factor)
                                          for _ in range(num_layers)])
        self.output = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](self.reduces[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1)))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out