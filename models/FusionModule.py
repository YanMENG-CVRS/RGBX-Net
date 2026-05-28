import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        x1 = self.q1(x1)
        x2 = self.q2(x2)
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CSFusion(nn.Module):
    def __init__(self, dim, reduction=1):
        super(CSFusion, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, self.dim * 2, kernel_size=1),
                    nn.Sigmoid())
        self.proj = nn.Conv2d(self.dim, self.dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.dim)

    def forward(self, x1, x2):
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        B, C, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        w = self.mlp(x).reshape(B, 2, C, H, W).permute(1, 0, 2, 3, 4)  # 2 B C H W
        return self.relu(self.bn(self.proj((x1 * w[0] + x2 * w[1]))))  # B C H W


class FeatureFusionBlock(nn.Module):
    def __init__(self, dim, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.proj_e = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.cross_attn = CrossAttention(dim, num_heads=num_head)
        self.LFA = CSFusion(dim)

    def forward(self, x, x_e):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x_e = x_e.permute(0, 2, 3, 1)

        x = self.norm(x)
        x_e = self.norm_e(x_e)

        fused = self.LFA(x, x_e)

        x = x.reshape(B, H * W, C)
        x_e = x_e.reshape(B, H * W, C)
        x1, x_e1 = self.cross_attn(x, x_e)

        x1 = x1.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_e1 = x_e1.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x = self.proj(torch.cat([x1, fused], dim=1))
        x_e = self.proj_e(torch.cat([x_e1, fused], dim=1))

        return x, x_e
