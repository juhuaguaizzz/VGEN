import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
# class ImageProjModel(nn.Module):
#     #对图像进行了解构clip_extra_context_tokens
#     def __init__(self,cross_attention_dim=1536, clip_embeddings_dim=1280, clip_extra_context_tokens=24,dtype=None,
#                  num_buckets=50):
#         super().__init__()
#         self.padding=nn.Parameter(torch.randn(1,256,cross_attention_dim - clip_embeddings_dim,dtype=torch.float16))
#     def forward(self,last_hidden_states):
#
#         return torch.cat([last_hidden_states, self.padding.expand(last_hidden_states.shape[0],-1,-1)], dim=-1)

class ImageProjModel(nn.Module):
    # 对图像进行了解构clip_extra_context_tokens
    def __init__(self, cross_attention_dim=1536, clip_embeddings_dim=1280, clip_extra_context_tokens=24, dtype=None,
                 num_buckets=50):
        super().__init__()

        #原版
        self.text_proj = nn.Linear(1024, 256, bias=False).to(torch.float16)

        # 新版 attn-like
        self.trans_seq = nn.Sequential(
            nn.Linear(256,512).to(torch.float16),
            nn.ReLU(),
            nn.Linear(512,256).to(torch.float16)
        )

        # self.trans_to_1536 = nn.Linear(256, 1536).to(torch.float16)

    # def forward(self, last_hidden_states, text_features,origin_pics=None):
    #     dtype=torch.float16
    #
    #     text_features=text_features.to(dtype)  #文本语义向量
    #     patch_tokens = last_hidden_states.to(dtype)
    #     #做一个维度对齐
    #     text_proj = torch.cat((text_features, self.text_proj(text_features)), dim=-1) #[batchsize,L,1280]
    #
    #     text_proj = F.normalize(text_proj, dim=-1)  # Normalize text
    #     patch_tokens = F.normalize(patch_tokens, dim=-1)  # Normalize image
    #
    #     score_map = torch.bmm(patch_tokens, text_proj.unsqueeze(dim=2)).squeeze(dim=2)
    #
    #
    #     fused_score = self.trans_seq(score_map)
    #
    #     return last_hidden_states,fused_score


    #VGEN 使用clip最后一层
    def forward(self, last_hidden_states, text_features,origin_pics=None):
        dtype=torch.float32

        return last_hidden_states,None




    # def forward(self, last_hidden_states, text_features,origin_pics=None):
    #     dtype=torch.float16
    #
    #     text_features=text_features.to(dtype)  #文本语义向量
    #     patch_tokens = last_hidden_states.to(dtype)
    #     #做一个维度对齐
    #     text_proj = torch.cat((text_features, self.text_proj(text_features)), dim=-1) #[batchsize,L,1280]
    #
    #     text_proj = F.normalize(text_proj, dim=-1)  # Normalize text
    #     patch_tokens = F.normalize(patch_tokens, dim=-1)  # Normalize image
    #
    #     score_map = torch.bmm(patch_tokens, text_proj.unsqueeze(dim=2)).squeeze(dim=2)
    #
    #     fused_score = self.trans_to_1536(score_map)
    #
    #     # fused_score = self.trans_seq(score_map)
    #
    #     return last_hidden_states,fused_score




    #旧版
    # def forward(self, last_hidden_states, text_features,origin_pics=None):
    #     text_features=text_features.to(torch.float16)
    #
    #     text_proj = torch.cat((text_features, self.text_proj(text_features)), dim=-1)
    #
    #     text_proj = text_proj / text_proj.norm(dim=-1, keepdim=True)  # [1, 1280]
    #     patch_tokens = last_hidden_states / last_hidden_states.norm(dim=-1, keepdim=True)  # [1,256, 1280]
    #
    #     # 点积：每个 patch 和文本语义向量的余弦相似度
    #     text_proj = text_proj.unsqueeze(-1)
    #     patch_tokens = patch_tokens.to(torch.float16)
    #     score_clip = torch.bmm(patch_tokens, text_proj).squeeze(-1)  # [256]
    #
    #     score_clip = (score_clip - score_clip.min()) / (score_clip.max() - score_clip.min() + 1e-6)
    #
    #     return last_hidden_states,score_clip