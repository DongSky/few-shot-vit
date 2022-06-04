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
        local_classifier_args = {'in_dim': self.encoder.out_dim, 'n_classes': int(classifier_args['n_classes']+1)}#classifier_args
        #local_classifier_args['n_classes'] += 1
        self.classifier = models.make(classifier, **classifier_args)
        self.classifier_local = models.make(classifier, **local_classifier_args)
    
    def forward(self, x, is_teacher=False):
        x, x1 = self.encoder(x)
        x_reshape = x.permute(0, 2, 3, 1)
        if not is_teacher:
            y_reshape = self.classifier_local(x_reshape)
        else:
            y_reshape = self.classifier(x_reshape)
        #y_reshape = self.classifier1(x_reshape)
        y_token = y_reshape.permute(0, 3, 1, 2)
        # y_token_argmax = torch.argmax(y_token, dim=1)
        y = self.classifier(x1)
        # y_argmax = torch.argmax(y, dim=1)
        return y_token, y, x1

@register('token-label-ep')
class TokenLabelOfflineEpisodic(nn.Module):
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        #self.classifier = models.make(classifier, **classifier_args)
        #self.classifier_local = models.make(classifier, **classifier_args)
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
        feat_shot = feat_query.view(*shot_shape, x_shot.shape[-1], -1).transpose(-1, -2) # bxnxkxhwxc
        b, q, t, c = feat_query.size()
        b, n, k, t, c = feat_shot.size()
        feat_query = feat_query.contiguous().view(b, q, 1, t, c).expand(-1, -1, n, -1, -1)
        feat_shot = feat_shot.contiguous().view(b, 1, n, k*t, c).expand(-1, q, -1, -1, -1)
        feat_sim = torch.cosine_similarity(feat_query, feat_shot, dim=-1)
        feat_sim_ = torch.topk(feat_sim, 1, dim=-1)[0].mean(-1) # b, q, n, k
        logits = feat_sim_.mean(-1) # b, q, n

        x_shot = x_shot.mean(dim=-2)
        x_query = x_query.view(x_query.size(0), x_query.size(1)*x_query.size(2), -1)
        x_shot = F.normalize(x_shot, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        cls_logits = utils.compute_logits(
                x_query, x_shot, metric='dot', temp=self.temp)
        return logits, cls_logits
