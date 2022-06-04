import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from torch import optim as optim
from timm.scheduler import CosineLRScheduler
from timm.optim import AdamW

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
#import scipy.stats
#def mean_confidence_interval(data, confidence=0.95):
#    a = 1.0 * np.array(data)
#    n = len(a)
#    se = scipy.stats.sem(a)
#    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
#    return h

## token labeling revision:
## 1.topk (k=3 or 5) one-hot label, then use softmax to generate soft target (using soft cross-entropy loss to supervise patch token)
## 2.teacher model use global classifier 
##
class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

#def generate_softlabel(logits, smoothing=0.1, k=3, device='cuda'):
#    n_classes = logits.size(1)
#    off_value = smoothing / n_classes
#    on_value = 1 - smoothing + off_value
#    logits = logits.permute(0, 2, 3, 1).view(-1, n_classes)
#    value, idx = logits.topk(k)
#
#    soft_label = torch.full(logits.size(), off_value, device=device).scatter_(1, idx, on_value)
#    return soft_label

def generate_softlabel(logits, smoothing=0.1, k=3, bp=10, device='cuda'):
    n_classes = logits.size(1)
    off_value = smoothing / n_classes
    on_value = 1 - smoothing + off_value
    logits_max, _ = logits.max(dim=1, keepdim=True)
    b, c, h, w = logits_max.size()
    logits_max = logits_max.view(b, c, h*w)
    _, pos_select = logits_max.topk(h*w - bp, dim=-1)
    pos_mask = torch.zeros_like(logits_max).scatter(-1, pos_select, 1)
    pos_mask = pos_mask.permute(0, 2, 1).view(-1, 1)
    #pos_select = logits_max.topk(h*w - bp)
    #pos_select = pos_select.permute(0, 2, 3, 1).view(-1, 1)
    logits = logits.permute(0, 2, 3, 1).view(-1, n_classes)
    value, idx = logits.topk(k)
    bg_map = torch.full(pos_mask.size(), c, device=device)

    soft_label = torch.full((logits.size(0), logits.size(1)+1), off_value, device=device).scatter_(1, idx, on_value)
    soft_bg_label = torch.full((logits.size(0), logits.size(1)+1), off_value, device=device).scatter_(1, bg_map, on_value)
    soft_label = soft_label * pos_mask + soft_bg_label * (1 - pos_mask)
    return soft_label

def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))
    print(config)
    bp = 10 if config['bg_token_num'] is None else int(config['bg_token_num'])
    #### Dataset ####
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('n_train_shot') is not None:
        n_train_shot = config['n_train_shot']
    else:
        n_train_shot = n_shot
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1
    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True,
                              num_workers=8, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(
            train_dataset[0][0].shape, len(train_dataset),
            train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)
    # train for meta learning phase
    train_batches = len(train_loader)
    # train_meta_dataset = datasets.make(config['train_dataset'],
    #                               **config['train_dataset_args'])
    # utils.log('train meta dataset: {} (x{}), {}'.format(
    #         train_meta_dataset[0][0].shape, len(train_meta_dataset),
    #         train_meta_dataset.n_classes))
    # if config.get('visualize_datasets'):
    #     utils.visualize_dataset(train_meta_dataset, 'train_meta_dataset', writer)
    # train_meta_sampler = CategoriesSampler(
    #         train_meta_dataset.label, train_batches,
    #         n_train_way, n_train_shot + n_query,
    #         ep_per_batch=ep_per_batch)
    # train_meta_loader = DataLoader(train_meta_dataset, batch_sampler=train_meta_sampler,
    #                           num_workers=8, pin_memory=True)

    # val
    # if config.get('val_dataset'):
    #     eval_val = True
    #     val_dataset = datasets.make(config['val_dataset'],
    #                                 **config['val_dataset_args'])
    #     val_loader = DataLoader(val_dataset, config['batch_size'],
    #                             num_workers=8, pin_memory=True)
    #     utils.log('val dataset: {} (x{}), {}'.format(
    #             val_dataset[0][0].shape, len(val_dataset),
    #             val_dataset.n_classes))
    #     if config.get('visualize_datasets'):
    #         utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    # else:
    #     eval_val = False
    eval_val = True
    val_dataset = datasets.make(config['val_dataset'],
                                **config['val_dataset_args'])
    utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    val_sampler = CategoriesSampler(
            val_dataset.label, 200,
            n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True

        fs_dataset = datasets.make(config['fs_dataset'],
                                   **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(
                fs_dataset[0][0].shape, len(fs_dataset),
                fs_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        n_way = 5
        n_query = 15
        n_shots = [1, 5]
        fs_loaders = []
        for n_test_shot in n_shots:
            fs_sampler = CategoriesSampler(
                    fs_dataset.label, 200,
                    n_way, n_test_shot + n_query, ep_per_batch=4)
            fs_loader = DataLoader(fs_dataset, batch_sampler=fs_sampler,
                                   num_workers=8, pin_memory=True)
            fs_loaders.append(fs_loader)
    else:
        eval_fs = False

    ########

    #### Model and Optimizer ####
    model = models.make(config['model'], **config['model_args'])
    teacher = models.make(config['model'], **config['model_args'])
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model_sv['model_args'] = config['model_args']
        model_sv['model'] = config['model']
        teacher = models.load(model_sv)
    else:
        teacher = models.make(config['model'], **config['model_args'])
    
    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    #### model EMA as teacher ####
    # ema = utils.ModelEma(model, decay=0.997)



    utils.log('num params: {}'.format(utils.compute_n_params(model)))
    if config['model_args']['encoder'].startswith('res'):
        optimizer, lr_scheduler = utils.make_optimizer(
                model.parameters(),
                config['optimizer'], **config['optimizer_args'])
    else:
        lr = float(config['optimizer_args']['lr']) * (config['batch_size']/512)
        optimizer = AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=float(config['optimizer_args']['weight_decay']))
        lr_scheduler = CosineLRScheduler(optimizer, warmup_lr_init=float(config['optimizer_args']['warmup_lr']), t_initial=config['max_epoch'], cycle_decay=0.1, warmup_t=int(config['optimizer_args']['warmup']))
    ########
    
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    tl_weight = config['token_label_weight'] if config['token_label_weight'] is not None else 0.5
    tl_soft_k = config['tl_soft_k'] if config['tl_soft_k'] is not None else 3
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()
    criterion_TL = SoftTargetCrossEntropy()
    teacher.eval()
    for epoch in range(1, max_epoch + 1 + 1):
        if epoch == max_epoch + 1:
            if not config.get('epoch_ex'):
                break
            train_dataset.transform = train_dataset.default_transform
            train_loader = DataLoader(
                    train_dataset, config['batch_size'], shuffle=True,
                    num_workers=8, pin_memory=True)

        timer_epoch.s()
        aves_keys = ['tl', 'ta', 'vl', 'va']
        if eval_fs:
            for n_test_shot in n_shots:
                aves_keys += ['fsa-' + str(n_test_shot)]
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, weak_data, label in tqdm(train_loader, desc='train', leave=False):
            data, weak_data, label = data.cuda(), weak_data.cuda(), label.cuda()
            #### fetch query-support images from category sampler ####
            # meta_data, meta_cls_label = next(iter(train_meta_loader))
            # meta_data, meta_cls_label = meta_data.cuda(), meta_cls_label.cuda()
            #### supervised learning phase, use std model only ####
            logits_token, logits, token = model(data)
            cls_loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)
            #### local alignment based meta learning phase,
            #### use both std model (for query images) to generate both global token and local tokens
            #### and ema model (for support images) (currently, only global token is essential)
            # x_shot, x_query = fs.split_shot_query(
            #         meta_data, n_train_way, n_train_shot, n_query,
            #         ep_per_batch=ep_per_batch)
            # supp_label = fs.make_nk_label(n_train_way, n_train_shot, ep_per_batch=ep_per_batch).cuda()
            # supp_label_onehot = fs.make_nway_kshot_onehot_label(n_train_way, n_train_shot, n_train_shot, ep_per_batch=ep_per_batch).cuda()
            # query_label = fs.make_nk_label(n_train_way, n_query,
            #         ep_per_batch=ep_per_batch).cuda()

            # b, n, q, c, h, w = x_query.size()
            # b, n, k, c, h, w = x_shot.size()
            out = model.module.encoder.out_dim
            #### inference of query images ####
            # query_logits_token, query_logits, query_token = model(x_query.view(b*n*q, c, h, w))
            with torch.no_grad():
                logits_token_t, logits_t, token_t = teacher(weak_data, True)#ema.module(data, True)
                soft_label = generate_softlabel(logits_token_t, k=tl_soft_k, bp=bp)


            #### self promoted token labeling ####
            b, c, h, w = logits_token_t.size()
            # centerness or sharpen, currently omitted
            logits_flatten = logits_token.permute(0, 2, 3, 1).view(-1, c+1)
            token_loss = criterion_TL(logits_flatten, soft_label)
            # pixel level kl loss 
            # kl_loss = utils.softmax_kl_loss(logits_token, logits_token_t) / (h*w)
            loss = cls_loss + 0.5 * token_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.update(model)

            aves['tl'].add(loss.item())
            aves['ta'].add(acc)

            logits = None; loss = None

        # eval
        if eval_val:
            model.eval()
            va_lst = []
            for data, label in tqdm(val_loader, desc='val', leave=False):
                # data, label = data.cuda(), label.cuda()
                x_shot, x_query = fs.split_shot_query(
                            data.cuda(), n_way, n_shot, n_query, ep_per_batch=ep_per_batch)
                fs_label = fs.make_nk_label(
                        n_way, n_query, ep_per_batch=ep_per_batch).cuda()
                # with torch.no_grad():
                    # logits = model(data)
                    # loss = F.cross_entropy(logits, label)
                    # acc = utils.compute_acc(logits, label)
                    
                with torch.no_grad():
                    b, n, q, c, h, w = x_query.size()
                    b, n, k, c, h, w = x_shot.size()
                    # logits = fs_model(x_shot, x_query).view(-1, n_way)
                    query_logits_token, query_logits, query_token = model(x_query.view(b*n*q, c, h, w))
                    support_logits_token, support_logits, support_token = model(x_shot.view(b*n*k, c, h, w))
                    query_token = query_token.view(b, n*q, out)
                    support_token = support_token.view(b, n, k, out).mean(dim=2)
                    global_logits = utils.compute_logits(query_token, support_token, metric='cos', temp=10.0).view(-1, n_way)
                    global_loss = F.cross_entropy(global_logits, fs_label)
                    acc = utils.compute_acc(global_logits, fs_label)
                    #va_lst.append(acc)
                
                aves['vl'].add(global_loss.item())
                aves['va'].add(acc)
            #acc_std = mean_confidence_interval(va_lst) * 100

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            va_lst = []
            for i, n_test_shot in enumerate(n_shots):
                np.random.seed(0)
                for data, _ in tqdm(fs_loaders[i],
                                    desc='fs-' + str(n_test_shot), leave=False):
                    x_shot, x_query = fs.split_shot_query(
                            data.cuda(), n_way, n_test_shot, n_query, ep_per_batch=4)
                    label = fs.make_nk_label(
                            n_way, n_query, ep_per_batch=4).cuda()
                    with torch.no_grad():
                        b, n, q, c, h, w = x_query.size()
                        b, n, k, c, h, w = x_shot.size()
                        # logits = fs_model(x_shot, x_query).view(-1, n_way)
                        query_logits_token, query_logits, query_token = model(x_query.view(b*n*q, c, h, w))
                        support_logits_token, support_logits, support_token = model(x_shot.view(b*n*k, c, h, w))
                        query_token = query_token.view(b, n*q, out)
                        support_token = support_token.view(b, n, k, out).mean(dim=2)
                        global_logits = utils.compute_logits(query_token, support_token, metric='cos', temp=10.0).view(-1, n_way)
                        global_loss = F.cross_entropy(global_logits, label)
                        acc = utils.compute_acc(global_logits, label)
                        #va_lst.append(acc)

                    aves['fsa-' + str(n_test_shot)].add(acc)
            #acc_std = mean_confidence_interval(va_lst) * 100

        # post
        if lr_scheduler is not None:
            lr_scheduler.step(epoch-1) # 1-based epochs

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        if epoch <= max_epoch:
            epoch_str = str(epoch)
        else:
            epoch_str = 'ex'
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(
                epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_test_shot in n_shots:
                key = 'fsa-' + str(n_test_shot)
                log_str += ' {}: {:.4f}'.format(n_test_shot, aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        if epoch <= max_epoch:
            torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))

            if (save_epoch is not None) and epoch % save_epoch == 0:
                torch.save(save_obj, os.path.join(
                    save_path, 'epoch-{}.pth'.format(epoch)))

            if aves['va'] > max_va:
                max_va = aves['va']
                torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
        else:
            torch.save(save_obj, os.path.join(save_path, 'epoch-ex.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)


