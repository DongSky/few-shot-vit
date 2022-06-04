import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class TokenQKV(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        #self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        #self.k = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        #self.v = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
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
        # x_support: BxNxKxC -> Bx(NK)xC
        # x_query: BxNxQx(HW)xC -> BxNxQx(HW)xC
        b, n, k, c = x_support.size()
        h = self.num_heads
        c_split = c // h
        q, hw = x_query.size(2), x_query.size(3)
        x_support = x_support.view(b, n*k, c)
        q_support = x_support#self.q(x_support)
        k_query = x_query#self.k(x_query)
        v_query = x_query#self.v(x_query)
        o_support = x_support#self.proj(self.v(x_support)) #x_support -> v_support -> o_support
        q_support = q_support.view(b, n, 1, k, h, c_split).permute(0, 2, 4, 1, 3, 5).view(b, 1, h, n*k, c_split)
        k_query = k_query.view(b, n, q, hw, h, c_split).permute(0, 1, 2, 4, 3, 5).view(b, n*q, h, hw, c_split)
        v_query = v_query.view(b, n, q, hw, h, c_split).permute(0, 1, 2, 4, 3, 5).view(b, n*q, h, hw, c_split)
        attn = (q_support @ k_query.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        # b, nq, h, nk, c_split
        o_support = o_support.view(b, 1, n*k, c)
        o_query = (attn @ v_query).permute(0, 1, 3, 2, 4).reshape(b, n*q, n*k, c).contiguous()
        #o_query = self.proj(o_query)
        #o_query = self.proj_drop(o_query)
        return o_query, o_support

class TokenQKV_Params(nn.Module):
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
        # x_support: BxNxKxC -> Bx(NK)xC
        # x_query: BxNxQx(HW)xC -> BxNxQx(HW)xC
        b, n, k, c = x_support.size()
        h = self.num_heads
        c_split = c // h
        q, hw = x_query.size(2), x_query.size(3)
        x_support = x_support.view(b, n*k, c)
        q_support = x_support#self.q(x_support)
        k_query = x_query#self.k(x_query)
        v_query = x_query#self.v(x_query)
        o_support = x_support#self.proj(self.v(x_support)) #x_support -> v_support -> o_support
        q_support = q_support.view(b, n, 1, k, h, c_split).permute(0, 2, 4, 1, 3, 5).view(b, 1, h, n*k, c_split)
        k_query = k_query.view(b, n, q, hw, h, c_split).permute(0, 1, 2, 4, 3, 5).view(b, n*q, h, hw, c_split)
        v_query = v_query.view(b, n, q, hw, h, c_split).permute(0, 1, 2, 4, 3, 5).view(b, n*q, h, hw, c_split)
        attn = (q_support @ k_query.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        #attn = self.attn_drop(attn)
        # b, nq, h, nk, c_split
        o_support = o_support.view(b, 1, n*k, c)
        o_query = (attn @ v_query).permute(0, 1, 3, 2, 4).reshape(b, n*q, n*k, c).contiguous()
        #o_query = self.proj(o_query)
        #o_query = self.proj_drop(o_query)
        return o_query, o_support

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


@register('meta-token')
class MetaToken(nn.Module):

    def __init__(self, encoder, classifier, classifier_args, encoder_args={}, num_heads=1, qkv_bias=True, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.transformer = TokenQKV(self.encoder.out_dim, num_heads, qkv_bias=qkv_bias)
        self.classifier = nn.Linear(self.encoder.out_dim, classifier_args['n_classes'])
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        #feat_tot, x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        #feat_shot, feat_query = feat_tot[:len(x_shot)], feat_tot[-len(x_query):]
        #x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        feat_shot, x_shot = self.encoder(x_shot)
        feat_query, x_query= self.encoder(x_query)
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        feat_query = feat_query.view(*query_shape, x_shot.shape[-1], -1).transpose(-1, -2)
        o_query, o_shot = self.transformer(feat_query, x_shot)
        # x_shot = x_shot.view(*shot_shape, -1)
        # x_query = x_query.view(*query_shape, -1)
        #cls_logits = self.classifier(x_tot)

        if self.method == 'cos':
            # x_shot = x_shot.mean(dim=-2)
            #o_shot = F.normalize(o_shot, dim=-1)
            #o_query = F.normalize(o_query, dim=-1)
            metric = 'cos'
        elif self.method == 'sqr':
            # x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits_kshot(
                o_query, o_shot, metric=metric, temp=self.temp)
        x_shot = x_shot.mean(dim=-2)
        x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)
        return logits, cls_logits

# 1. we consider a sampling strategy for support features
# HowTo: given GAP support token as well as feature without GAP, compute attention between support token and support feature. finally, select top-k% (e.g., k=50)

class Sampling(nn.Module):
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

        gathered_token = selected_attn @ selected_feat
        return gathered_token.reshape(b1, n1, k1, c).contiguous()

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

@register('meta-token-v2')
class MetaTokenV2(nn.Module):

    def __init__(self, encoder, classifier, classifier_args, encoder_args={}, num_heads=1, qkv_bias=True, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.transformer = TokenQKV(self.encoder.out_dim, num_heads, qkv_bias=qkv_bias)
        self.sampling = Sampling()
        self.classifier = nn.Linear(self.encoder.out_dim, classifier_args['n_classes'])
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        #feat_tot, x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        #feat_shot, feat_query = feat_tot[:len(x_shot)], feat_tot[-len(x_query):]
        #x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        feat_shot, x_shot = self.encoder(x_shot)
        feat_query, x_query= self.encoder(x_query)
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        feat_query = feat_query.view(*query_shape, x_shot.shape[-1], -1).transpose(-1, -2)
        feat_shot = feat_shot.view(*shot_shape, x_shot.shape[-1], -1).transpose(-1, -2)
        x_shot = self.sampling(x_shot, feat_shot)
        o_query, o_shot = self.transformer(feat_query, x_shot)
        # x_shot = x_shot.view(*shot_shape, -1)
        # x_query = x_query.view(*query_shape, -1)
        #cls_logits = self.classifier(x_tot)

        if self.method == 'cos':
            # x_shot = x_shot.mean(dim=-2)
            #o_shot = F.normalize(o_shot, dim=-1)
            #o_query = F.normalize(o_query, dim=-1)
            metric = 'cos'
        elif self.method == 'sqr':
            # x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits_kshot(
                o_query, o_shot, metric=metric, temp=self.temp)
        x_shot = x_shot.mean(dim=-2)
        x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)
        return logits, cls_logits

@register('meta-token-v3')
class MetaTokenV3(nn.Module):

    def __init__(self, encoder, classifier, classifier_args, encoder_args={}, num_heads=1, qkv_bias=True, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.transformer = TokenQKV_Local(self.encoder.out_dim, num_heads, qkv_bias=qkv_bias)
        self.sampling = SamplingKeepToken()
        self.classifier = nn.Linear(self.encoder.out_dim, classifier_args['n_classes'])
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        #feat_tot, x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        #feat_shot, feat_query = feat_tot[:len(x_shot)], feat_tot[-len(x_query):]
        #x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        feat_shot, x_shot = self.encoder(x_shot)
        feat_query, x_query= self.encoder(x_query)
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        feat_query = feat_query.view(*query_shape, x_shot.shape[-1], -1).transpose(-1, -2)
        feat_shot = feat_shot.view(*shot_shape, x_shot.shape[-1], -1).transpose(-1, -2)
        selected_x_shot = self.sampling(x_shot, feat_shot)
        selected_x_query = self.sampling(x_query, feat_query)
        o_query, o_shot = self.transformer(selected_x_query, selected_x_shot)
        # x_shot = x_shot.view(*shot_shape, -1)
        # x_query = x_query.view(*query_shape, -1)
        #cls_logits = self.classifier(x_tot)

        if self.method == 'cos':
            # x_shot = x_shot.mean(dim=-2)
            #o_shot = F.normalize(o_shot, dim=-1)
            #o_query = F.normalize(o_query, dim=-1)
            metric = 'cos'
        elif self.method == 'sqr':
            # x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits_local_kshot(
                o_query, o_shot, metric=metric, temp=self.temp)
        x_shot = x_shot.mean(dim=-2)
        x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)
        return logits, cls_logits