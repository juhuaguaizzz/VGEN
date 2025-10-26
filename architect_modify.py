import random
from typing import Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import Tensor



class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads


        # VGEN正常模型
        # 定义 QKV 映射，手动控制 Q, K, V 计算
        # self.k_linear = nn.Linear(1280, embed_dim, bias=False)
        # self.v_linear = nn.Linear(1280, embed_dim, bias=False)

        #VGEN使用最后一层CLIP
        self.k_linear = nn.Linear(1024, embed_dim, bias=False)
        self.v_linear = nn.Linear(1024, embed_dim, bias=False)

        self.scale_linear = nn.Linear(2*256,1)
        self.seq = nn.Sequential(self.scale_linear,nn.Softplus())

        nn.init.constant_(self.scale_linear.weight, 0.05)  # 或其他你觉得合理的 scale
        nn.init.constant_(self.scale_linear.bias, 7.5)  # 初始输出接近 7.5


    def forward(self, query, key, value,act_scale=1.0,clip_score=None,attention_mask=None):

        if key.shape[-1]==1024:
            key = key.unsqueeze(dim=1).expand(-1, 256, -1)
            value = value.unsqueeze(dim=1).expand(-1, 256, -1)

        # query_imp=self.net(query)
        q_channels = query.size(1)
        if len(key.shape)!=3: #就是4
            k_channels = key.size(2)
            k_batch=key.size(0)*key.size(1)
        else:
            k_channels = key.size(1)
            k_batch = key.size(0)



        # 手动计算 Q, K, V
        Q = query  # (seq_len, batch_size, embed_dim)
        K = self.k_linear(key)  # (seq_len, batch_size, embed_dim)
        V = self.v_linear(value)  # (seq_len, batch_size, embed_dim)

        if self.num_heads ==1:

            # 计算缩放点积注意力
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))

            clip_score = clip_score.unsqueeze(1).expand(-1, scores.shape[1], -1)

            # fused_score = self.scale_linear(torch.cat([scores, clip_score], dim=-1)) * clip_score

            scores = scores + clip_score*act_scale

            attn_weights = torch.nn.functional.softmax(scores, dim=-1)

            output = torch.matmul(attn_weights, V)

        else:

            # 将 Q, K, V 拆分成多个头
            Q = Q.view(Q.size(0), q_channels, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(k_batch, k_channels, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(k_batch, k_channels, self.num_heads, self.head_dim).transpose(1, 2)

            # 计算缩放点积注意力
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))

            if clip_score is not None:

                clip_score = clip_score.unsqueeze(1).expand(-1, scores.shape[2], -1)

                clip_score = clip_score.unsqueeze(1).expand(scores.shape[0], self.num_heads, -1, -1)
                # print(scores.shape,clip_score.shape)
                fused_score=self.scale_linear(torch.cat([scores,clip_score],dim=-1))*clip_score

                scores = scores+fused_score*act_scale

            else:
                pass

            attn_weights = torch.nn.functional.softmax(scores, dim=-1)

            # 计算注意力输出
            output = torch.matmul(attn_weights, V)

            # 合并多个头的输出
            output = output.transpose(1, 2).contiguous().view(query.size(0), q_channels, -1)


        return output, attn_weights


class sd3_adapter_attn_processor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, hidden_size=1536, cross_attention_dim=1536, scale=0.75,split_context_dim=154,text_scale=1.0):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.text_scale=text_scale
        self.scale = scale  # 用于控制力道
        self.split_context_dim = split_context_dim

        self.sd3_to_multihead_attn = CustomMultiHeadAttention(hidden_size,num_heads=24)

        # self.sd3_to_ln = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-5)

        self.sd3_to_adaptive_weight_layer = nn.Linear(hidden_size * 2, 1)


    def __call__(
            self,
            attn,
            hidden_states: torch.FloatTensor,  # 噪声图像
            encoder_hidden_states: torch.FloatTensor = None,  # 文本编码 + 图像编码
            attention_mask=None,
            *args,
            **kwargs,
    ) -> tuple[tuple[Tensor | Any, Tensor], Tensor | Any]:
        inner_dim = 1536
        head_dim = inner_dim // attn.heads

        residual = hidden_states  # 保存当前的输入

        # 分割(batch_size,257,1536)
        pics_hidden_states = encoder_hidden_states[1]
        clip_score = encoder_hidden_states[2]
        encoder_hidden_states = encoder_hidden_states[0]

        # 获取批次大小
        batch_size = encoder_hidden_states.shape[0]

        #1 196 1536
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)


        # 文本经过qkv映射
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        multi_output, multi_attn = self.sd3_to_multihead_attn(query, pics_hidden_states, pics_hidden_states,
                                                              clip_score=clip_score)

        # 图像和文本做拼接，然后实现交叉注意力计算
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)



        # 转成这个维度
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)


        # 计算 scaled_dot_product_attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :residual.shape[1]],
            hidden_states[:, residual.shape[1]:],
        )

        weighted_input = torch.cat([multi_output, hidden_states], dim=-1)  # 拼接两个特征
        adaptive_weight = torch.sigmoid(self.sd3_to_adaptive_weight_layer(weighted_input))  # 生成一个权重系数

        add_inf=adaptive_weight * multi_output

        # add_inf = 1 * multi_output

        hidden_states = self.scale * 0.65 * add_inf + hidden_states  # 残差


        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)



        # 最后的输出
        if not attn.context_pre_only:
            # 残差连接 - 最终加权
            # encoder_hidden_states = self.scale * encoder_add + encoder_hidden_states
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # cross_attn_residual=self.sd3_to_out(cross_attn_output)
        return (hidden_states,0.0), encoder_hidden_states


