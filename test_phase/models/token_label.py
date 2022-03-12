import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# @register('classifier')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


# @register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)

@register('token-label')
class TokenLabelOffline(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        #classifier_args['n_classes'] = 64
        self.classifier = models.make(classifier, **classifier_args)
        local_classifier_args = classifier_args
        local_classifier_args['n_classes'] += 1
        #self.classifier = models.make(classifier, **classifier_args)
        self.classifier_local = models.make(classifier, **local_classifier_args)
    
    def forward(self, x, is_teacher=False):
        x, x1 = self.encoder(x)
        out_dim = self.encoder.out_dim
        x = x / (float(out_dim) ** 0.5)
        x1 = x1 / (float(out_dim) ** 0.5)
        x_reshape = x.permute(0, 2, 3, 1)
        if not is_teacher:
            y_reshape = self.classifier_local(x_reshape)
        else:
            y_reshape = self.classifier(x_reshape)
        y_token = y_reshape.permute(0, 3, 1, 2)
        # y_token_argmax = torch.argmax(y_token, dim=1)
        y = self.classifier(x1)
        # y_argmax = torch.argmax(y, dim=1)
        return y_token, y, x1

def l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))

@register('token-label-ep')
class TokenLabelOfflineEpisodic(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.classifier_local = models.make(classifier, **classifier_args)
        self.temp = 10.0
    
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
        _, c, h, w = feat_shot.size()
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        feat_query = feat_query.view(*query_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxqxhwxc
        feat_shot = feat_shot.view(*shot_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxnxkxhwxc
        b, q, t, c = feat_query.size()
        b, n, k, t, c = feat_shot.size()
        feat_query = feat_query.contiguous().view(b, q, 1, t, c).expand(-1, -1, n, -1, -1)
        feat_shot = feat_shot.contiguous().view(b, 1, n, k*t, c).expand(-1, q, -1, -1, -1)
        feat_sim = torch.cosine_similarity(feat_query, feat_shot, dim=-1)
        feat_sim_ = torch.topk(feat_sim, 1, dim=-1)[0].mean(-1) # b, q, n, k
        logits = feat_sim_.mean(-1) # b, q, n

        x_shot = x_shot.mean(dim=-2)
        # x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)
        return logits, cls_logits

class SelfReweight(nn.Module):
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
        attn = attn.sum(dim=-1, keepdim=True)
        attn = torch.sigmoid(attn)
        return attn * feature
        # attn = torch.sigmoid(attn, dim=-1)
        # selected_attn, selected_idx = attn.topk(dim=-1, k=int(hw*self.sampling_rate), largest=True)
        # selected_attn = selected_attn / selected_attn.sum(dim=-1, keepdim=True)
        # b3, nk, q_, tk = selected_attn.size()
        # selected_idx_reshape = selected_idx.reshape(b3, nk, tk, 1).contiguous().expand(b3, nk, tk, c)
        # selected_feat = feature.gather(2, selected_idx_reshape)
        # selected_attn = selected_attn.view(b3, nk, tk, 1)
        # gathered_token = selected_attn * selected_feat
        # return gathered_token.reshape(b1, n1, k1, int(hw*self.sampling_rate), c).contiguous()

class MetaLearner(nn.Module):
    def __init__(self, dim=384, ratio=4, num_classes=64):
        super().__init__()
        self.intra_task_learner = nn.Sequential(
            nn.Linear(dim, dim*ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim*ratio, dim, bias=True),
        )
        self.intra_class_learner = nn.Sequential(
            nn.Linear(dim, dim*ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim*ratio, dim, bias=True),
        )
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(dim, num_classes, bias=True)
        self.classifier_local = nn.Linear(dim, num_classes, bias=True)
    
    def forward(self, query, support, is_train=True):
        if is_train:
            pass
        else:
            pass

@register('token-label-ep-rw')
class TokenLabelOfflineEpisodicReweight(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.classifier_local = models.make(classifier, **classifier_args)
        self.intra_task_learner = nn.Sequential(
            nn.Linear(dim, dim*ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim*ratio, dim, bias=True),
        )
        self.intra_class_learner = nn.Sequential(
            nn.Linear(dim, dim*ratio, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim*ratio, dim, bias=True),
        )
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.temp = 10.0
    
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
        _, c, h, w = feat_shot.size()
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        feat_query = feat_query.view(*query_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxqxhwxc
        feat_shot = feat_shot.view(*shot_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxnxkxhwxc

        b, q, t, c = feat_query.size()
        b, n, k, t, c = feat_shot.size()

        feat_task_shot = feat_shot.view(b, n*k, 1, t, c).mean(dim=1, keepdim=True).mean(dim=3, keepdim=True)#b111c
        feat_class_shot = feat_shot.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).transpose(1,2)#b1n1c
        weight_task_shot = self.intra_task_learner(feat_task_shot) # kernel to generate spatial saliency map
        weight_class_shot = self.intra_class_learner(feat_class_shot)
        weight_class_shot = torch.sigmoid(weight_class_shot) # channel reweighting

        
        feat_query = feat_query.contiguous().view(b, q, 1, t, c).expand(-1, -1, n, -1, -1)
        feat_shot = feat_shot.contiguous().view(b, 1, n, k*t, c).expand(-1, q, -1, -1, -1)

        # feat_query_channel_reweight = (feat_query * weight_class_shot.expand_as(feat_query) + feat_query) / 2
        # feat_shot_channel_reweight = (feat_shot * weight_class_shot.expand_as(feat_shot) + feat_shot) / 2
        feat_query_channel_reweight = feat_query * weight_class_shot.expand_as(feat_query)
        feat_shot_channel_reweight = feat_shot * weight_class_shot.expand_as(feat_shot)

        feat_channel_map_query = feat_query_channel_reweight @ weight_task_shot.transpose(-1, -2)
        feat_channel_map_shot = feat_shot_channel_reweight @ weight_task_shot.transpose(-1, -2)

        feat_channel_map_query = torch.sigmoid(feat_channel_map_query)
        feat_channel_map_shot = torch.sigmoid(feat_channel_map_shot)

        feat_query_channel_reweight = feat_query_channel_reweight * feat_channel_map_query.expand_as(feat_query_channel_reweight)
        feat_shot_channel_reweight = feat_shot_channel_reweight * feat_channel_map_shot.expand_as(feat_shot_channel_reweight)


        feat_sim = torch.cosine_similarity(feat_query, feat_shot, dim=-1)
        feat_sim_ = torch.topk(feat_sim, 1, dim=-1)[0].mean(-1) # b, q, n, k
        logits = feat_sim_.mean(-1) # b, q, n

        feat_sim_reweight = torch.cosine_similarity(feat_query, feat_shot, dim=-1)
        feat_sim_reweight_ = torch.topk(feat_sim, 1, dim=-1)[0].mean(-1) # b, q, n, k
        logits = feat_sim_.mean(-1) # b, q, n
        logits_reweight = feat_sim_reweight_.mean(-1)

        x_shot_reweight = l2norm(feat_shot_channel_reweight.mean(3), dim=-1)#bqnc
        x_query_reweight = l2norm(feat_query_channel_reweight.mean(3), dim=-1)#bqnc
        cls_logits_reweight = torch.einsum("bqnc,bqnc->bqn", [x_query_reweight, x_shot_reweight])


        x_shot = x_shot.mean(dim=-2)
        # x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)

        
        return logits, logits_reweight, cls_logits, cls_logits_reweight

@register('token-label-ep-cr')
class TokenLabelOfflineEpisodic(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.classifier_local = models.make(classifier, **classifier_args)
        self.temp = 10.0
    
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
        _, c, h, w = feat_shot.size()
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)
        feat_query = feat_query.view(*query_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxqxhwxc
        feat_shot = feat_shot.view(*shot_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxnxkxhwxc
        b, q, t, c = feat_query.size()
        b, n, k, t, c = feat_shot.size()
        feat_query = feat_query.contiguous().view(b, q, 1, t, c).expand(-1, -1, n, -1, -1)
        feat_shot = feat_shot.contiguous().view(b, 1, n, k*t, c).expand(-1, q, -1, -1, -1)

        feat_query_full = feat_query.expand(-1, -1, -1, k*t, -1)

        channel_attn = (feat_query_full.transpose(-1, -2) @ feat_shot) / (k*t+0.0)**0.5
        channel_attn = channel_attn.softmax(dim=-1) # b,q,n,c,c
        feat_query = feat_query @ channel_attn.transpose(-1, -2)
        feat_sim = torch.cosine_similarity(feat_query, feat_shot, dim=-1)
        feat_sim_ = torch.topk(feat_sim, 1, dim=-1)[0].mean(-1) # b, q, n, k
        logits = feat_sim_.mean(-1) # b, q, n

        x_shot = x_shot.mean(dim=-2)
        # x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)
        return logits, cls_logits    
    # def forward(self, x, is_teacher=False):
    #     x, x1 = self.encoder(x)
    #     out_dim = self.encoder.out_dim
    #     x = x / (float(out_dim) ** 0.5)
    #     x1 = x1 / (float(out_dim) ** 0.5)
    #     x_reshape = x.permute(0, 2, 3, 1)
    #     if not is_teacher:
    #         y_reshape = self.classifier_local(x_reshape)
    #     else:
    #         y_reshape = self.classifier(x_reshape)
    #     y_token = y_reshape.permute(0, 3, 1, 2)
    #     # y_token_argmax = torch.argmax(y_token, dim=1)
    #     y = self.classifier(x1)
    #     # y_argmax = torch.argmax(y, dim=1)
    #     return y_token, y, x1

@register('token-label-v2')
class TokenLabelOffline(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args, dim=128):
        super().__init__()
        self.dim = dim
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.projection = nn.Sequential([
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.ReLU(True),
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.ReLU(True),
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim)
        ])
        self.classifier_local = nn.Sequential([
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.ReLU(True),
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.ReLU(True),
            nn.Linear(self.encoder.out_dim, self.dim)
        ])#models.make(classifier, **classifier_args)
    
    def forward(self, x):
        x, x1 = self.encoder(x)
        out_dim = self.encoder.out_dim
        x = x / (float(out_dim) ** 0.5)
        x1 = x1 / (float(out_dim) ** 0.5)
        x_reshape = x.permute(0, 2, 3, 1)
        y_projection = self.projection(x_reshape)
        y_reshape = self.classifier_local(y_projection)
        y_token = y_reshape.permute(0, 3, 1, 2)
        # y_token_argmax = torch.argmax(y_token, dim=1)
        y = self.classifier(x1)
        # y_argmax = torch.argmax(y, dim=1)
        return y_token, y, x1, y_projection.permute(0, 3, 1, 2)

class TokenLabelOfflineV2Pretrain(nn.Module):
    def __init__(self):
        super().__init__()

class TokenLabelOfflineV2Episodic(nn.Module):
    def __init__(self):
        super().__init__()
