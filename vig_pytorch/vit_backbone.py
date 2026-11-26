"""
Vision Transformer backbone for the ViG-branch ablation.

Default config (dim=384, depth=12 -> ~21.7M parameters) is sized to match
vig_s (channels=320, 16 blocks -> 21.39M), so ViT can replace the ViG branch
in two_branch.py (`--vig-backbone vit`) for an equal-parameter ablation.

Exposes the same interface as the ViG branch:

- forward_stage_features(x) -> 3 feature maps [B, C, H, W] taken at 1/3,
  2/3 and 3/3 of the block stack (cls token dropped, tokens reshaped back
  to the patch grid), matching the ViG forward_stage_features output used
  by the multi-stage graph distance regularizer.
- forward(x) -> logits when built with num_classes (vig_raw mode), otherwise
  the pooled cls feature.
"""

import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be positive and divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        num_classes: int = None,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.dim = dim
        self.depth = depth
        self.grid = img_size // patch_size
        num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes is not None else None

        # block indices (1-based) whose outputs become the 3 stage feature maps
        first = max(1, depth // 3)
        second = max(first + 1, 2 * depth // 3)
        self.stage_cuts = sorted({first, second, depth})[-3:]

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, dim]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.pos_drop(x)

    def _tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        # drop cls token and reshape the patch sequence back to [B, dim, H, W]
        B, N, C = tokens.shape
        patch = tokens[:, 1:, :]
        return patch.transpose(1, 2).reshape(B, C, self.grid, self.grid)

    def forward_stage_features(self, x: torch.Tensor) -> list:
        """Return 3 stage feature maps [B, dim, grid, grid] (ResNet-style stages)."""
        x = self._embed(x)
        stage_tokens = []
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x)
            if i in self.stage_cuts:
                stage_tokens.append(x)
        stage_tokens[-1] = self.norm(stage_tokens[-1])
        return [self._tokens_to_map(t) for t in stage_tokens]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if self.head is not None:
            return self.head(features)
        return features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embed(x)
        for blk in self.blocks:
            x = blk(x)
        cls = self.norm(x)[:, 0]
        return cls


if __name__ == "__main__":
    net = ViTBackbone(num_classes=100)
    data = torch.randn(2, 3, 224, 224)
    logits = net(data)
    maps = net.forward_stage_features(data)
    print("logits:", logits.shape)
    for i, m in enumerate(maps):
        print(f"stage {i + 1}:", m.shape)
    print("Total parameters:", sum(p.numel() for p in net.parameters()))
