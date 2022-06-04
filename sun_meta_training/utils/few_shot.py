import torch
import torch.nn.functional as F

def split_shot_query(data, way, shot, query, ep_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(ep_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous()#.view(ep_per_batch, way * query, *img_shape)
    return x_shot, x_query


def make_nk_label(n, k, ep_per_batch=1):
    # n: way number
    # k: query number
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(ep_per_batch)
    return label

def make_nway_kshot_onehot_label(n, k, q, ep_per_batch=1):
    # n: way number
    # k: k shot number
    # q: query number
    label = torch.arange(n).unsqueeze(1).expand(n, q).reshape(-1)
    onehot = F.one_hot(label, num_classes=n)
    onehot = onehot.unsqueeze(2).expand(n*q, n, k)
    onehot = onehot.reshape(n*q, n*k)
    labels = onehot.repeat(ep_per_batch, 1)
    return labels
