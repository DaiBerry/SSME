# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block, ParallelScalingBlock


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """ 
    def __init__(self, global_pool=None, qk_scale=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        depth = 8 # 自注意力块的个数
        # embed_dim = 1024 # 或者是512
        img_size  = 128
        patch_size = 16
        in_chans = 3
        depth = 16
        embed_dim = 768
        mlp_ratio=4
        qkv_bias = True
        num_heads = 12
        
        self.blocks_atten = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        x = x.unsqueeze(0) # x=1,3,128,128
        B = x.shape[0]
        x = self.patch_embed(x) # x=1,64,1024;64个patchs

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1) # x=1,65,1024
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            y = blk(x) # x : 1,65,1024
            flag = 0
            if flag == 1 :
                x = x + y
                flag -= 1
            else :
                x = y
                flag += 1
            

        self.atten = x[:, 1:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            
            x = self.norm(x) # x=1,65,1024
            
            outcome = x[:, 0] # outcome=1,1024 第一维全取：　第二维度取索引为０
            #outcome = x
        # print("featrue size : {}", outcome.shape)

        return outcome
    
    def self_atten(self,x):
        # apply Transformer blocks
        for blk in self.blocks_atten:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]
        outcome = self.forward_head(x)
        return outcome

        


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), class_token = False, **kwargs) # block_fn=ParallelScalingBlock,
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model