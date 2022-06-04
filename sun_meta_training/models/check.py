import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class SamplingKeepToken(nn.Module):
    def __init__(self, sampling_rate=0.5):
        super().__init__()
        self.sampling_rate = sampling_rate
    
    def forward(self, token, feature):
        # token: BxNxKxC -> Bx(NK)x1xC
        # feature: BxNxKx(HW)xC -> Bx(NK)x(HW)xC
        b1, n1, k1, c = token.size()
        token = token.view(b1, n1*k1, 1, c)
        b2, n2, k2, hw, c = feature.size()
        feature = feature.view(b2, n2*k2, hw, c)
        scale = c ** -0.5
        attn = (token @ feature.transpose(-1, -2)) * scale
        attn = attn.softmax(dim=-1)
        selected_attn, selected_idx = attn.topk(dim=-1, k=int(hw*self.sampling_rate), largest=True)
        selected_attn = selected_attn / selected_attn.sum(dim=-1, keepdim=True)
        b3, nk, q_, tk = selected_attn.size()
        selected_idx_reshape = selected_idx.reshape(b3, nk, tk, 1).contiguous().expand(b3, nk, tk, c)
        selected_feat = feature.gather(2, selected_idx_reshape)
        selected_attn = selected_attn.view(b3, nk, tk, 1)
        gathered_token = selected_attn * selected_feat
        return gathered_token.reshape(b1, n1, k1, int(hw*self.sampling_rate), c).contiguous()

class TokenQKV_Local(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        #self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(embed_dim, embed_dim)
        #self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x_query, x_support):
        # x_support: BxNxKx(HW)xC -> Bx(NK)(HW)xC
        # x_query: BxNxQx(HW)xC -> BxNxQx(HW)xC
        b, n, k, hw, c = x_support.size()
        h = self.num_heads
        c_split = c // h
        q, hw = x_query.size(2), x_query.size(3)
        x_support = x_support.view(b, n, k*hw, c)
        q_support = self.q(x_support)
        k_query = self.k(x_query)
        v_query = self.v(x_query)
        o_support = self.v(x_support) #x_support -> v_support -> o_support
        q_support = q_support.view(b, n, 1, k*hw, h, c_split).permute(0, 2, 4, 1, 3, 5).view(b, 1, h, n*k*hw, c_split)
        k_query = k_query.view(b, n, q, hw, h, c_split).permute(0, 1, 2, 4, 3, 5).view(b, n*q, h, hw, c_split)
        v_query = v_query.view(b, n, q, hw, h, c_split).permute(0, 1, 2, 4, 3, 5).view(b, n*q, h, hw, c_split)
        attn = (q_support @ k_query.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        # b, nq, h, nk, c_split
        o_support = o_support.view(b, 1, n*k, hw, c).expand(b, n*q, n*k, hw, c)
        o_query = (attn @ v_query).permute(0, 1, 3, 2, 4).reshape(b, n*q, n*k, hw, c).contiguous()
        #o_query = self.proj(o_query)
        #o_query = self.proj_drop(o_query)
        return o_query, o_support

if __name__ == "__main__":
    module = TokenQKV_Local(7, 1, True)
    a = torch.rand(3, 2, 2, 8, 7)
    b = torch.rand(3, 2, 5, 8, 7)
    c, d = module(b, a)
    print(c.size(), d.size())
    sim = F.cosine_similarity(c, d, dim=-1).mean(dim=-1)
    print(sim.size())
    print(sim)
    