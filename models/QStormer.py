import torch
from torch import nn, einsum
from torchvision import transforms
from torch.nn import functional as F

from torchsummary import summary

from einops import rearrange

from PIL import Image

import matplotlib.pyplot as plt


class MDTA(nn.Module):
    def __init__(self, channels, num_heads, return_q=False, takes_q=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.return_q = return_q
        self.takes_q = takes_q
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        if takes_q:
            self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
            self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        else:
            self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
            self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, q_in=None):
        b, c, h, w = x.shape
        if self.takes_q:
            k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1)
            k = k.reshape(b, self.num_heads, -1, h * w)
            v = v.reshape(b, self.num_heads, -1, h * w)

            k = F.normalize(k, dim=-1)
            if q_in==None:
                print("MDTA didn't recieve Q in forward pass, replacing with V.")
                q = F.normalize(v, dim=-1)
            else:
                q = q_in
        else:
            q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
            q = q.reshape(b, self.num_heads, -1, h * w)
            k = k.reshape(b, self.num_heads, -1, h * w)
            v = v.reshape(b, self.num_heads, -1, h * w)
            
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        # print(f'\t\tq_shape:', q.shape) #TEST
        # print(f'\t\tk_shape:', k.shape) #TEST
        # print(f'\t\tv_shape:', v.shape) #TEST

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        # print(f'\t\tattn:', attn.shape) #TEST
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        # print(f'\t\toutput:', out.shape) #TEST
        if self.return_q:
            return out, q
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
    def __init__(self, channels, num_heads, expansion_factor, return_q=False, takes_q=False):
        super(TransformerBlock, self).__init__()
        self.return_q = return_q
        self.channels = channels #TEST
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads, return_q=return_q, takes_q=takes_q)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x, q_in=None):
        b, c, h, w = x.shape
        if q_in == None and self.return_q:
            att, q = self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w))
            x = att + x
        elif q_in == None and not self.return_q:
            # print('\t\tnorm', x.shape) #TEST
            # print('\t\tchannel', self.channels) #TEST
            x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w))
        elif q_in != None and self.return_q:
            att, q = self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w), q_in)
            x = att + x
        else:
            x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                            .contiguous().reshape(b, c, h, w), q_in)
        
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        
        if self.return_q:
            return x, q
        else:
            return x
        

class TransformerBulk(nn.Module):
    def __init__(self, num_blocks, channels, num_heads, expansion_factor, return_q=False, takes_q=False):
        super().__init__()

        self.return_q = return_q
        self.num_blocks = num_blocks

        self.bulk = nn.ModuleList()
        self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, False, takes_q))
        for _ in range(1, num_blocks - 1):
            self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, False, False))
        self.bulk.append(TransformerBlock(channels, num_heads, expansion_factor, return_q, False))

    def forward(self, x, q_in=None):
        if q_in==None:
            for idx in range(self.num_blocks - 1):
                # print(f'\ttransformer {idx}') #TEST
                x = self.bulk[idx](x)
            # print(f'\ttransformer {self.num_blocks}') #TEST
            if self.return_q:
                x, q = self.bulk[-1](x)
                return x, q
            else:
                return self.bulk[-1](x)
        else:
            x = self.bulk[0](x, q_in)
            for idx in range(1, self.num_blocks - 1):
                # print(f'\ttransformer {idx}') #TEST
                x = self.bulk[idx](x)
            # print(f'\ttransformer {self.num_blocks}') #TEST
            if self.return_q:
                x, q = self.bulk[-1](x)
                return x, q
            else:
                return self.bulk[-1](x)
            

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
        # print('yes', x.shape)
        # print('mamade', self.ubody(x).shape)
        return self.ubody(x)
    

class QStormer(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_layers=4,
                 expansion_factor=2.66):
        super(QStormer, self).__init__()

        enc_retq = ([True] * (num_layers-1))
        enc_retq.append(False)

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([TransformerBulk(num_tb, num_ch, num_ah, expansion_factor, retq) for num_tb, num_ah, num_ch, retq in
                                       zip(num_blocks, num_heads, channels, enc_retq)])

        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1

        # self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
        #                               for i in reversed(range(2, len(channels)))])

        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([TransformerBulk(num_blocks[2], channels[2], num_heads[2], expansion_factor, False, True)])
        self.decoders.append(TransformerBulk(num_blocks[1], channels[1], num_heads[1], expansion_factor, False, True))
        self.decoders.append(TransformerBulk(num_blocks[0], channels[0], num_heads[0], expansion_factor, False))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[0], num_heads[0], expansion_factor)
                                          for _ in range(num_layers)])
        self.output = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        
        # print('encoder 1') #TEST
        out_enc1, q1 = self.encoders[0](fo)
        # print('encoder 2') #TEST
        out_enc2, q2 = self.encoders[1](self.downs[0](out_enc1))
        # print('encoder 3') #TEST
        out_enc3, q3 = self.encoders[2](self.downs[1](out_enc2))
        # print('encoder 4') #TEST
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        # print('decoder 3') #TEST
        out_dec3 = self.decoders[0](self.ups[0](out_enc4), q3)
        # print('decoder 2') #TEST
        out_dec2 = self.decoders[1](self.ups[1](out_dec3), q2)
        # print('dec 2 out shape:', out_dec2.shape) #TEST
        # print('decoder 1') #TEST
        fd = self.decoders[2](self.ups[2](out_dec2))
        # print('refiner') #TEST
        fr = self.refinement(fd)
        out = self.output(fr)
        return out