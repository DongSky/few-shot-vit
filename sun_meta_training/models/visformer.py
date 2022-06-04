import torch
import torch.nn as nn
from einops import rearrange
# from weight_init import to_2tuple, trunc_normal_

import torch
import math
import warnings
import collections
from itertools import repeat
from .models import register
DEBUG = True
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

__all__=[
    'visformer_small', 'visformer_tiny', 'visformer_tiny_80', 'net5', 'net5_80'
]

def drop_path(x, drop_prob:float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)


class BatchNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., group=8, spatial_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.spatial_conv = spatial_conv
        if self.spatial_conv:
            if group < 2: #net setting
                hidden_features = in_features * 5 // 6
            else:
                hidden_features = in_features * 2
        self.hidden_features = hidden_features
        self.group = group
        self.drop = nn.Dropout(drop)
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0, bias=False)
        self.act1 = act_layer()
        if self.spatial_conv:
            self.conv2 = nn.Conv2d(hidden_features, hidden_features, 3, stride=1, padding=1,
                                   groups=self.group, bias=False)
            self.act2 = act_layer()
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop(x)

        if self.spatial_conv:
            x = self.conv2(x)
            x = self.act2(x)

        x = self.conv3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim_ratio=1., qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, stride=1, padding=0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.qkv(x)
        qkv = rearrange(x, 'b (x y z) h w -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        if DEBUG:
            self.attn_store = attn.clone().cpu().detach()
        attn = self.attn_drop(attn)
        x = attn @ v

        x = rearrange(x, 'b y (h w) z -> b (y z) h w', h=H, w=W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)
def norm_layer(planes):
    return nn.BatchNorm2d(planes)
class ConvBlock(nn.Module):

    def __init__(self, inplanes, hidden_planes, planes):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(inplanes, hidden_planes, 3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(hidden_planes)
        self.conv2 = conv3x3(hidden_planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 3, stride=2, padding=1, bias=False),norm_layer(planes))
        # self.last = last
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        # if not self.last:
        out = self.maxpool(out)

        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim_ratio=1., mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm,
                 group=8, attn_disabled=False, spatial_conv=False):
        super().__init__()
        self.attn_disabled = attn_disabled
        self.spatial_conv = spatial_conv
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if not attn_disabled:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, head_dim_ratio=head_dim_ratio, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       group=group, spatial_conv=spatial_conv) # new setting

    def forward(self, x):
        if not self.attn_disabled:
            x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_pe = norm_layer is not None
        if self.norm_pe:
            self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) does not match model ({self.img_size[1]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.norm_pe:
            x = self.norm(x)
        return x


class Visformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, init_channels=32, num_classes=1000, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=LayerNorm, attn_stage='111', pos_embed=True, spatial_conv='111',
                 vit_embedding=False, group=8, pool=True, conv_init=False, embedding_norm=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.out_dim = embed_dim * 2 if vit_embedding == False else embed_dim
        self.init_channels = init_channels
        self.img_size = img_size
        self.vit_embedding = vit_embedding
        self.pool = pool
        self.conv_init = conv_init
        if isinstance(depth, list) or isinstance(depth, tuple):
            self.stage_num1, self.stage_num2, self.stage_num3 = depth
            depth = sum(depth)
        else:
            self.stage_num1 = self.stage_num3 = depth // 3
            self.stage_num2 = depth - self.stage_num1 - self.stage_num3
        self.pos_embed = pos_embed
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # stage 1
        if self.vit_embedding:
            self.using_stem = False
            self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=16, in_chans=3, embed_dim=embed_dim,
                                           norm_layer=embedding_norm)
            img_size //= 16
        else:
            if self.init_channels is None:
                self.using_stem = False
                self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=8, in_chans=3, embed_dim=embed_dim//2,
                                               norm_layer=embedding_norm)
                img_size //= 8
            else:
                self.using_stem = True
                self.stem = ConvBlock(3, init_channels, embed_dim // 2)
                self.patch_embed1 = nn.Identity()
                # self.stem = nn.Sequential(
                #     nn.Conv2d(3, self.init_channels, 7, stride=2, padding=3, bias=False),
                #     BatchNorm(self.init_channels),
                #     nn.ReLU(inplace=True)
                # )
                # img_size //= 2
                # self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=4, in_chans=self.init_channels,
                #                                embed_dim=embed_dim//2, norm_layer=embedding_norm)
                img_size //= 4

        if self.pos_embed:
            if self.vit_embedding:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))
            else:
                self.pos_embed1 = nn.Parameter(torch.zeros(1, embed_dim//2, img_size, img_size))
            self.pos_drop = nn.Dropout(p=drop_rate)
        self.stage1 = nn.ModuleList([
            Block(
                dim=embed_dim//2, num_heads=num_heads, head_dim_ratio=0.5, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[0] == '0'), spatial_conv=(spatial_conv[0] == '1')
            )
            for i in range(self.stage_num1)
        ])

        #stage2
        if not self.vit_embedding:
            self.patch_embed2 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim//2, embed_dim=embed_dim,
                                           norm_layer=embedding_norm)
            img_size //= 2
            if self.pos_embed:
                self.pos_embed2 = nn.Parameter(torch.zeros(1, embed_dim, img_size, img_size))
        self.stage2 = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[1] == '0'), spatial_conv=(spatial_conv[1] == '1')
            )
            for i in range(self.stage_num1, self.stage_num1+self.stage_num2)
        ])

        # stage 3
        if not self.vit_embedding:
            self.patch_embed3 = PatchEmbed(img_size=img_size, patch_size=2, in_chans=embed_dim, embed_dim=embed_dim*2,
                                           norm_layer=embedding_norm)
            img_size //= 2
            if self.pos_embed:
                self.pos_embed3 = nn.Parameter(torch.zeros(1, embed_dim*2, img_size, img_size))
        self.stage3 = nn.ModuleList([
            Block(
                dim=embed_dim*2, num_heads=num_heads, head_dim_ratio=1.0, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                group=group, attn_disabled=(attn_stage[2] == '0'), spatial_conv=(spatial_conv[2] == '1')
            )
            for i in range(self.stage_num1+self.stage_num2, depth)
        ])

        # head
        if self.pool:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        if not self.vit_embedding:
            self.norm = norm_layer(embed_dim*2)
            # self.head = nn.Linear(embed_dim*2, num_classes)
        else:
            self.norm = norm_layer(embed_dim)
            # self.head = nn.Linear(embed_dim, num_classes)

        # weights init
        if self.pos_embed:
            trunc_normal_(self.pos_embed1, std=0.02)
            if not self.vit_embedding:
                trunc_normal_(self.pos_embed2, std=0.02)
                trunc_normal_(self.pos_embed3, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if self.conv_init:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        if self.using_stem:
            x = self.stem(x)

        # stage 1
        x = self.patch_embed1(x)
        if self.pos_embed:
            x = x + self.pos_embed1
            x = self.pos_drop(x)
        for b in self.stage1:
            x = b(x)

        # stage 2
        if not self.vit_embedding:
            x = self.patch_embed2(x)
            if self.pos_embed:
                x = x + self.pos_embed2
                x = self.pos_drop(x)
        for b in self.stage2:
            x = b(x)

        # stage3
        if not self.vit_embedding:
            x = self.patch_embed3(x)
            if self.pos_embed:
                x = x + self.pos_embed3
                x = self.pos_drop(x)
        for b in self.stage3:
            x = b(x)

        # head
        x = self.norm(x)
        if self.pool:
            x1 = self.global_pooling(x)
        else:
            x1 = x[:, :, 0, 0]

        # x = self.head( x.view(x.size(0), -1) )
        return x, x1.view(x1.size(0), -1)


def visformer_tiny(**kwargs):
    model = Visformer(img_size=224, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model
# @register('visformer_tiny_80')
def visformer_tiny_80(**kwargs):
    model = Visformer(img_size=80, init_channels=16, embed_dim=192, depth=[7,4,4], num_heads=3, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model

def visformer_small(**kwargs):
    model = Visformer(img_size=224, init_channels=32, embed_dim=384, depth=[7,4,4], num_heads=6, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model
@register('visformer_micro_80')
def visformer_small_80(**kwargs):
    model = Visformer(img_size=80, init_channels=64, embed_dim=256, depth=[4,2,3], num_heads=6, mlp_ratio=4., group=8,
                      attn_stage='011', spatial_conv='100', norm_layer=BatchNorm, conv_init=True,
                      embedding_norm=BatchNorm, **kwargs)
    return model

def net5(**kwargs):
    model = Visformer(init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., group=1, attn_stage='111',
                      spatial_conv='111', vit_embedding=False, norm_layer=BatchNorm, conv_init=True, **kwargs)
    return model

def net5_80(**kwargs):
    model = Visformer(img_size=80, init_channels=32, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., group=1, attn_stage='111',
                      spatial_conv='111', vit_embedding=False, norm_layer=BatchNorm, conv_init=True, **kwargs)
    return model


if __name__ == '__main__':
    torch.manual_seed(0)
    inputs = torch.rand(2, 3, 80, 80)

    net = visformer_small_80()

    parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of parameters:{}'.format(parameters))
    x = net(inputs)
    print(x.shape)
