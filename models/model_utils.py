import torch.nn as nn
import torch
import math


# //---before unet---//

class SDB(nn.Module):
    def __init__(self, in_ch, out_ch, deep=4, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )
        if deep >= 2:
            self.block2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
        )
        if deep >= 3:
            self.block3 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )
        if deep >= 4:
            self.block4 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )
        if deep >=5:
            self.block5 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )
        if deep >= 6:
            self.block6 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )
        if deep == 1:
            self.my_Super_resnet = self.Super_resnet1
        elif deep == 2:
            self.my_Super_resnet = self.Super_resnet2
        elif deep == 3:
            self.my_Super_resnet = self.Super_resnet3
        elif deep == 4:
            self.my_Super_resnet = self.Super_resnet4
        elif deep == 5:
            self.my_Super_resnet = self.Super_resnet5
        elif deep == 6:
            self.my_Super_resnet = self.Super_resnet6
        else:
            raise RuntimeError('error deep:', deep)
    
    def Super_resnet1(self, x):
        return self.block1(x)
    
    def Super_resnet2(self, x):
        out1 = self.block1(x)
        out1_scale = out1.mul(self.res_scale)
        out2 = self.block2(out1)
        return out1_scale+out2
    
    def Super_resnet3(self, x):
        out1 = self.block1(x)
        out1_scale = out1.mul(self.res_scale)
        out2 = self.block2(out1)
        out2_scale = out2.mul(self.res_scale)        
        out3 = self.block3(out1_scale+out2)
        return out1_scale+out2_scale+out3
    
    def Super_resnet4(self, x):
        out1 = self.block1(x)
        out1_scale = out1.mul(self.res_scale)
        out2 = self.block2(out1)
        out2_scale = out2.mul(self.res_scale)        
        out3 = self.block3(out1_scale+out2)
        out3_scale = out3.mul(self.res_scale)   
        out4 = self.block4(out1_scale+out2_scale+out3)
        return out1_scale+out2_scale+out3_scale+out4

    def Super_resnet5(self, x):
        out1 = self.block1(x)
        out1_scale = out1.mul(self.res_scale)
        out2 = self.block2(out1)
        out2_scale = out2.mul(self.res_scale)        
        out3 = self.block3(out1_scale+out2)
        out3_scale = out3.mul(self.res_scale)
        out4 = self.block4(out1_scale+out2_scale+out3)        
        out4_scale = out4.mul(self.res_scale)        
        out5 = self.block5(out1_scale+out2_scale+out3_scale+out4)
        return out1_scale+out2_scale+out3_scale+out4_scale+out5

    def Super_resnet6(self, x):
        out1 = self.block1(x)
        out1_scale = out1.mul(self.res_scale)
        out2 = self.block2(out1)
        out2_scale = out2.mul(self.res_scale)        
        out3 = self.block3(out1_scale+out2)
        out3_scale = out3.mul(self.res_scale)
        out4 = self.block4(out1_scale+out2_scale+out3)        
        out4_scale = out4.mul(self.res_scale)
        out5 = self.block5(out1_scale+out2_scale+out3_scale+out4)
        out5_scale = out5.mul(self.res_scale)
        out6 = self.block6(out1_scale+out2_scale+out3_scale+out4_scale+out5)
        
        return out1_scale+out2_scale+out3_scale+out4_scale+out5_scale+out6


    def forward(self, x):
        return self.my_Super_resnet(x)

# //---unet partI---//
class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.

    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.GroupNorm(num_groups, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
        self.block1 = block(in_ch, out_ch)
        self.block2 = block(out_ch, out_ch)

    def forward(self, x):
        out = self.block1(x)
        return self.block2(out) + out

class PSA(nn.Module):
    r"""
        Create global dependence.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_channles):
        super(self_attention, self).__init__()
        self.in_channels = in_channles

        self.f = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_channles, out_channels=in_channles // 8, kernel_size=1)
        self.h1 = nn.Conv2d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.h2 = nn.Conv2d(in_channels=in_channles, out_channels=in_channles, kernel_size=1)
        self.softmax_ = nn.Softmax(dim=2)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.init_weight(self.f)
        self.init_weight(self.g)
        self.init_weight(self.h1)
        self.init_weight(self.h2)

    def init_weight(self, conv):
        nn.init.kaiming_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def forward(self, x1, x2):
        batch_size, channels, height, width = x1.size()
        k_dim = channels//8
        # x1
        # k, q
        f = self.f(x1).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x1).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        # attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = torch.bmm(f, g) / math.sqrt(k_dim)
        attention = self.softmax_(attention)
        # v
        h = self.h1(x1).view(batch_size, channels, -1)  # B * C * (H * W)
        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W
        x1 = self.gamma1 * self_attention_map + x1

        # x2
        # k, q
        f = self.f(x2).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
        g = self.g(x2).view(batch_size, -1, height * width)  # B * C//8 * (H * W)

        # attention = torch.bmm(f, g)  # B * (H * W) * (H * W)
        attention = torch.bmm(f, g) / math.sqrt(k_dim)
        attention = self.softmax_(attention)
        # v
        h = self.h2(x2).view(batch_size, channels, -1)  # B * C * (H * W)
        self_attention_map = torch.bmm(h, attention).view(batch_size, channels, height, width)  # B * C * H * W
        x2 = self.gamma2 * self_attention_map + x2

        return x1, x2

class Bottleneck(nn.Module):
    def __init__(self, ch, num_groups):
        super().__init__()
        self.residual_block1 = Residual_block(ch, ch, num_groups)
        self.attention = AttentionBlock(ch,num_groups=num_groups)
        self.residual_block2 = Residual_block(ch, ch, num_groups)
    def forward(self, x):
        out = self.residual_block1(x)
        out = self.attention(out)
        return self.residual_block2(out)


class Down_block2(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels, time_emb_scale, num_groups):
        super().__init__()
        # 1.
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, in_ch),
            nn.SiLU(),
            nn.Linear(in_ch, in_ch)
        )
        # 2.
        self.residual_block = Residual_block( in_ch, out_ch, num_groups)
        # 3.
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = x + self.time_mlp(t)[:, :, None, None]
        # residual_block
        out = self.residual_block(x)
        # downsample
        out2 = self.downsample(out)
        return out, out2  # 采样前， 采样后

class Up_block2(nn.Module):
    def __init__(self, in_ch, out_ch, base_channels, time_emb_scale, num_groups):
        super().__init__()
        # 1.
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, in_ch),
            nn.SiLU(),
            nn.Linear(in_ch, in_ch)
        )
        # 2.
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        # 3.
        self.residual_block = Residual_block(in_ch*2, out_ch, num_groups)
        
    def forward(self, x, x_, t):
        # x: x_下采样，经过一系列操作得到的

        # add time
        x = x + self.time_mlp(t)[:, :, None, None]
        # upsample
        x = self.upsample(x)
        # connect
        x = torch.cat([x, x_], dim=1)
        # residual block  
        out = self.residual_block(x)

        return out
