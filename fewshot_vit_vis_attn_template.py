import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
import cv2

epochs = [i for i in range(5, 301, 5)]
DIR = "nest_3conv_3x3_mini_300epoch_recheck"
#DIR = "nest_3conv_2x_feat_res_cifarfs_300epoch"
PATH = os.path.join("save", DIR)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

mean = [[[0.485]], [[0.456]], [[0.406]]]
std = [[[0.229]], [[0.224]], [[0.225]]]

def main(config):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 4
    # batch_sampler = CategoriesSampler(
    #         dataset.label, n_batch, n_way, n_shot + n_query,
    #         ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=1, pin_memory=True)
    attn_list = []
    # model
    for E in range(1):
        # model_path = os.path.join(PATH, f"epoch-{E}.pth")
        if config.get('load') is None:
            model = models.make('meta-baseline', encoder=None)
        else:
            model = models.load(torch.load(config['load']))

        if config.get('load_encoder') is not None:
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder = encoder

        if config.get('_parallel'):
            model = nn.DataParallel(model)

        model.eval()
        #utils.log('num params: {}'.format(utils.compute_n_params(model)))

        # testing
        aves_keys = ['vl', 'va']
        aves = {k: utils.Averager() for k in aves_keys}

        test_epochs = args.test_epochs
        np.random.seed(0)
        va_lst = []
        i = 0
        for epoch in range(1, test_epochs + 1):
            for data, _ in loader:
                # x_shot, x_query = fs.split_shot_query(
                #         data.cuda(), n_way, n_shot, n_query,
                #         ep_per_batch=ep_per_batch)
                with torch.no_grad():
                    # print(data)
                    dense_logits = model.encoder(data.cuda())
                    # attn_list.append(model.encoder.levels[2].transformer_encoder[1].attn.attn_store)
                    b, c, h, w = dense_logits.shape
                    dense_logits = dense_logits.flatten(2).permute(0, 2 ,1)
                    cls_token = dense_logits.mean(dim=1, keepdim=True)
                    dense_logits = dense_logits.view(b, h*w, 16, c // 16).permute(0, 2, 1, 3)
                    cls_token = cls_token.view(b, 1, 16, c // 16).permute(0, 2, 1, 3)
                    attn = (cls_token @ dense_logits.transpose(-2, -1)) / ((c / 16) ** 0.5) # b 16, 1, N
                    attn = attn.softmax(-1)
                    # attn = model.encoder.levels[2].transformer_encoder[1].attn.attn_store
                    attn_mat = torch.mean(attn, dim=2) # reduce image block dim
                    attn_mat = torch.mean(attn_mat, dim=1) # reduce head dim
                    attn_mat = attn_mat.reshape(b, h, w)[0].detach().cpu().numpy()
                    attn_mat = cv2.resize((attn_mat-attn_mat.min()) / (attn_mat.max()-attn_mat.min()), (80, 80))[..., np.newaxis] ** 2
                    # print(attn_mat.max(), attn_mat.min())
                    attn_mat = cv2.applyColorMap((255*attn_mat[:, :, 0]).astype(np.uint8), cv2.COLORMAP_JET)
                    img = data.cpu()[0].numpy()
                    img = img * np.array(std) + np.array(mean)
                    img  = img * 255
                    # print(img)
                    img = img.transpose(1, 2, 0)[:, :, ::-1]
                    img = (img * 0.7 + attn_mat * 0.3).astype(np.uint8)
                    #img = (img * 0.5 + attn_mat * 0.45).astype(np.uint8)
                    cv2.imwrite(f"vis_colormap_shallow/sun/{i}.jpg", img)
                    i += 1
                    # if i == 10:
                    #     break
                    # residual_attn = torch.eye(attn_mat.size(1))
                    # aug_attn_mat = attn_mat + residual_attn
                    # aug_attn_mat = aug_attn_mat / aug_attn_mat.sum(dim=-1).unsqueeze(-1)
                    #print(model.encoder.levels[2].transformer_encoder[1].attn.attn_store.size())
    # for i in range(1, len(attn_list)):
    #     print(float((attn_list[i-1]-attn_list[i]).norm(p=2).item()))
            #     with torch.no_grad():
            #         if not args.sauc:
            #             logits = model(x_shot, x_query).view(-1, n_way)
            #             label = fs.make_nk_label(n_way, n_query,
            #                     ep_per_batch=ep_per_batch).cuda()
            #             loss = F.cross_entropy(logits, label)
            #             acc = utils.compute_acc(logits, label)

            #             aves['vl'].add(loss.item(), len(data))
            #             aves['va'].add(acc, len(data))
            #             va_lst.append(acc)
            #         else:
            #             x_shot = x_shot[:, 0, :, :, :, :].contiguous()
            #             shot_shape = x_shot.shape[:-3]
            #             img_shape = x_shot.shape[-3:]
            #             bs = shot_shape[0]
            #             p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
            #                     *shot_shape, -1).mean(dim=1, keepdim=True)
            #             q = model.encoder(x_query.view(-1, *img_shape)).view(
            #                     bs, -1, p.shape[-1])
            #             p = F.normalize(p, dim=-1)
            #             q = F.normalize(q, dim=-1)
            #             s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
            #             for i in range(bs):
            #                 k = s.shape[1] // 2
            #                 y_true = [1] * k + [0] * k
            #                 acc = roc_auc_score(y_true, s[i])
            #                 aves['va'].add(acc, len(data))
            #                 va_lst.append(acc)

            # print('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f} (@{})'.format(
            #         epoch, aves['va'].item() * 100,
            #         mean_confidence_interval(va_lst) * 100,
            #         aves['vl'].item(), _[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_classifier_mini_nest_micro.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=1)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)


