import torch
from torch import nn
import numpy as np


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride = 1,
        padding = 0,
        bias_attr=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()
    
    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out    

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=768,
        sub_num=2,
        patch_size=[4, 4],
        mode="pope",
    ):
        super().__init__()
        num_patches = (img_size[1] // (2**sub_num)) * (img_size[0] // (2**sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == "pope":
            self.proj = nn.Sequential(
                ConvBNLayer,)
    
    
    
    
    
    
    

class SVTRNet(nn.Module):
    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=[64, 128, 256],
        depth=[3,6,9],
        num_heads=[2,4,8],
        mixer=["Local"] * 6 + ["Global"] * 6, # Local atten, Global atten, Conv
        local_mixer=[[7, 11], [7, 11], [7, 11]],
        patch_merging="Conv",  # Conv, Pool, None
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer="nn.LayerNorm",
        sub_norm="nn.LayerNorm",
        epsilon=1e-6,
        out_channels=192,
        out_char_num=25,
        block_unit="Block",
        act="nn.GELU",
        last_stage=True,
        sub_num=2,
        prenorm=True,
        use_lenhead=False,
        **kwargs,
    ):
        super(SVTRNet, self).__init
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        parch_merging = (
            None if patch_merging != "Conv" and patch_merging != "Pool" else patch_merging
        )
        self.path_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num,
        )