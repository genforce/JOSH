import collections.abc
from itertools import repeat

import torch
import torch.nn as nn


class BasicLinearBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicLinearBlock, self).__init__()
        self.linear1 = nn.Linear(inplanes, planes)
        self.norm1 = nn.BatchNorm1d(planes)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(planes, planes)
        self.norm2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out += residual
        out = self.act(out)
        out = self.dropout(out)
        return out


class ROIFeatureRegressor(nn.Module):

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_channels=256,
    ):
        super(ROIFeatureRegressor, self).__init__()

        # Define the convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, conv_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels)

        self.conv2 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels)

        self.conv3 = nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels)

        # Flatten the output of the conv layers
        self.flatten = nn.Flatten()

        # Define fully connected layers with batch normalization
        self.fc1 = nn.Linear(conv_channels * 7 * 7, output_channels)
        self.bn_fc1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutional layers with batch normalization and ReLU activation
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten the features for fully connected layers
        x = self.flatten(x)

        # Apply fully connected layers with batch normalization and ReLU activation
        x = self.bn_fc1(self.fc1(x))

        return x


class MLPHead(nn.Module):

    def __init__(self, input_channels, output_channels, num_channels=None, num_basic_blocks=2, zero_init=False):
        super(MLPHead, self).__init__()
        head_layers = []
        if num_channels is None:
            num_channels = input_channels

        head_layers.append(nn.Sequential(nn.Linear(input_channels, num_channels), nn.BatchNorm1d(num_channels), nn.ReLU()))

        for _ in range(num_basic_blocks):
            head_layers.append(nn.Sequential(BasicLinearBlock(num_channels, num_channels)))
        if not zero_init:
            head_layers.append(nn.Linear(num_channels, output_channels))
        else:
            final_layer = nn.Linear(num_channels, output_channels, bias=False)
            nn.init.constant_(final_layer.weight, 0)
            head_layers.append(final_layer)
        self.head_layers = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head_layers(x)


def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_mem=True,
                 rope=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class HSITransformer(nn.Module):

    def __init__(self, embed_dim, num_heads, ff_dim, num_layers=1):
        super(HSITransformer, self).__init__()

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable Positional Embedding for 4 tokens
        self.pos_embedding = nn.Parameter(torch.randn(1, 4, embed_dim))

    def forward(self, human_token_1, human_token_2, scene_token_1, scene_token_2):
        # Stack tokens along sequence dimension (B, 4, C)
        tokens = torch.stack([human_token_1, human_token_2, scene_token_1, scene_token_2], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embedding

        # Pass through transformer encoder
        transformed_tokens = self.transformer_encoder(tokens)

        # Return the four updated tokens
        return transformed_tokens[:, 0, :], transformed_tokens[:, 1, :], transformed_tokens[:, 2, :], transformed_tokens[:, 3, :]


class RoPE2D(nn.Module):

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self._inv_freq_cache = {}

    def _get_inv_freq(self, D, device):
        if (D, device) not in self._inv_freq_cache:
            self._inv_freq_cache[D, device] = 1.0 / (self.base**(torch.arange(0, D, 2).float().to(device) / D))
        return self._inv_freq_cache[D, device]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, inv_freq):
        # pos1d: B, Seq (integer positions, can be negative)
        # Compute freqs directly from positions (like the CUDA kernel)
        freqs = pos1d.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)  # B, Seq, D/2
        freqs = torch.cat((freqs, freqs), dim=-1)  # B, Seq, D
        cos = freqs.cos()[:, None, :, :]  # B, 1, Seq, D
        sin = (self.F0 * freqs).sin()[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        inv_freq = self._get_inv_freq(D, tokens.device)
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], inv_freq)
        x = self.apply_rope1d(x, positions[:, :, 1], inv_freq)
        tokens = torch.cat((y, x), dim=-1)
        return tokens
