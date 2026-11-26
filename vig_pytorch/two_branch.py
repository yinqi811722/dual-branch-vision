import argparse
import math
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from backbone_adapters import RAW_ONLY_BACKBONES, build_pyramid_backbone, build_raw_classifier
from network.global_reasoning_unit import GloRe_Unit_2D
from vig import *
from vit_backbone import ViTBackbone


FUSION_MODES = ("fusion", "prob_fusion")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn_ab = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_ba = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        if feat_a.dim() == 2:
            feat_a = feat_a.unsqueeze(1)
            feat_b = feat_b.unsqueeze(1)

        attn_a, _ = self.cross_attn_ab(query=feat_a, key=feat_b, value=feat_b)
        attn_b, _ = self.cross_attn_ba(query=feat_b, key=feat_a, value=feat_a)
        fused = torch.cat([attn_a.mean(dim=1), attn_b.mean(dim=1)], dim=-1)
        return self.proj(fused)


class GloReResNet(nn.Module):
    """
    torchvision ResNet with GloRe reasoning units inserted after layer2/layer3
    (mirroring the insertion positions of conv3/conv4 in
    network/resnet50_2d_gcn_x5.py). Only used by the resnet_raw mode via
    --use-glore; the blocker BNs are zero-initialized so the network starts
    exactly as the plain ResNet and learns to use GloRe gradually.
    """

    def __init__(self, backbone: str, num_classes: int, mid_ratio: float = 0.5):
        super().__init__()
        if backbone == "50":
            model = models.resnet50(weights=None)
            layer2_dim, layer3_dim = 512, 1024
        else:
            model = models.resnet18(weights=None)
            layer2_dim, layer3_dim = 128, 256
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.resnet = model

        # units after layer2/layer3, num_mid scaled from the stage channels
        self.glore2 = GloRe_Unit_2D(num_in=layer2_dim, num_mid=int(layer2_dim * mid_ratio))
        self.glore3 = GloRe_Unit_2D(num_in=layer3_dim, num_mid=int(layer3_dim * mid_ratio))
        for unit in (self.glore2, self.glore3):
            nn.init.zeros_(unit.blocker.weight)
            nn.init.zeros_(unit.blocker.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x = r.layer1(x)
        x = self.glore2(r.layer2(x))
        x = self.glore3(r.layer3(x))
        x = r.layer4(x)
        x = r.avgpool(x)
        return r.fc(torch.flatten(x, 1))


class ViGResNetFusion(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        freeze_resnet: bool = False,
        freeze_vig: bool = False,
        proj_dim: int = 512,
        graph_dim: int = 128,
        graph_pool: int = 7,
        mode: str = "fusion",
        use_cross_attn: bool = False,
        resnet_backbone: str = "18",
        vig_backbone: str = "vig_s",
        use_glore: bool = False,
        glore_mid_ratio: float = 0.5,
        vit_dim: int = 384,
        vit_depth: int = 12,
        vit_num_heads: int = 6,
    ):
        super().__init__()
        self.mode = mode
        self.use_cross_attn = use_cross_attn
        self.graph_pool = graph_pool
        self.resnet_backbone = str(resnet_backbone)
        self.vig_backbone = str(vig_backbone)
        self.use_glore = use_glore
        self.glore_mid_ratio = glore_mid_ratio
        self.vit_dim = vit_dim
        self.vit_depth = vit_depth
        self.vit_num_heads = vit_num_heads

        if self.vig_backbone in RAW_ONLY_BACKBONES and self.mode != "vig_raw":
            raise ValueError(
                f"{self.vig_backbone} is a single-model comparison baseline and currently "
                "supports only --mode vig_raw"
            )

        if self.vig_backbone == "vit" and self.mode in FUSION_MODES and self.vit_depth < 3:
            raise ValueError(
                "Fusion with the ViT backbone requires vit_depth >= 3 "
                "so that three stage feature maps are available"
            )

        if self.mode in [*FUSION_MODES, "vig"]:
            self.vig, vig_dim, vig_stage_dims = self._build_vig_feature_backbone(self.vig_backbone)
            self.proj_vig = nn.Linear(vig_dim, proj_dim)
            self.norm_vig = nn.LayerNorm(proj_dim)
            self.vig_stage_projectors = self._make_stage_projectors(vig_stage_dims, graph_dim)
            if freeze_vig:
                self._freeze_module(self.vig)
        elif self.mode == "vig_raw":
            self.vig = self._build_vig_classifier(self.vig_backbone, num_classes)
            self.proj_vig = None
            self.norm_vig = None
            self.vig_stage_projectors = None
            if freeze_vig:
                self._freeze_module(self.vig)
        else:
            self.vig = None
            self.proj_vig = None
            self.norm_vig = None
            self.vig_stage_projectors = None

        if self.mode in [*FUSION_MODES, "resnet"]:
            self.resnet, res_dim, res_stage_dims = self._build_resnet_feature_backbone(self.resnet_backbone)
            self.proj_res = nn.Linear(res_dim, proj_dim)
            self.norm_res = nn.LayerNorm(proj_dim)
            self.res_stage_projectors = self._make_stage_projectors(res_stage_dims, graph_dim)
            if freeze_resnet:
                self._freeze_module(self.resnet)
        elif self.mode == "resnet_raw":
            if self.use_glore:
                self.resnet = GloReResNet(self.resnet_backbone, num_classes, mid_ratio=self.glore_mid_ratio)
            else:
                self.resnet = self._build_resnet_classifier(self.resnet_backbone, num_classes)
            self.proj_res = None
            self.norm_res = None
            self.res_stage_projectors = None
            if freeze_resnet:
                self._freeze_module(self.resnet)
        else:
            self.resnet = None
            self.proj_res = None
            self.norm_res = None
            self.res_stage_projectors = None

        if self.mode in FUSION_MODES:
            if self.use_cross_attn:
                fusion_dim = proj_dim
                self.fusion = CrossAttentionFusion(dim=proj_dim, num_heads=4)
            else:
                fusion_dim = proj_dim * 2
                self.fusion = None
        else:
            fusion_dim = proj_dim
            self.fusion = None

        if self.mode not in ["vig_raw", "resnet_raw"]:
            self.classifier = self._make_classifier(fusion_dim, num_classes)
        else:
            self.classifier = None

        if self.mode == "prob_fusion":
            self.vig_classifier = self._make_classifier(proj_dim, num_classes)
            self.res_classifier = self._make_classifier(proj_dim, num_classes)
        else:
            self.vig_classifier = None
            self.res_classifier = None

    def model_config(self, num_classes: int) -> Dict[str, object]:
        """Return the constructor arguments needed to restore this model."""
        projection = self.proj_vig if self.proj_vig is not None else self.proj_res
        projectors = self.vig_stage_projectors if self.vig_stage_projectors is not None else self.res_stage_projectors
        return {
            "num_classes": int(num_classes),
            "proj_dim": projection.out_features if projection is not None else 512,
            "graph_dim": (
                projectors[0][0].out_channels if projectors is not None else 128
            ),
            "graph_pool": int(self.graph_pool),
            "mode": self.mode,
            "use_cross_attn": bool(self.use_cross_attn),
            "resnet_backbone": self.resnet_backbone,
            "vig_backbone": self.vig_backbone,
            "use_glore": bool(self.use_glore),
            "glore_mid_ratio": float(self.glore_mid_ratio),
            "vit_dim": int(self.vit_dim),
            "vit_depth": int(self.vit_depth),
            "vit_num_heads": int(self.vit_num_heads),
        }

    @staticmethod
    def _freeze_module(module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _make_classifier(input_dim: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

    @staticmethod
    def _make_stage_projectors(stage_dims: List[int], graph_dim: int) -> nn.ModuleList:
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, graph_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(graph_dim),
                nn.ReLU(inplace=True),
            )
            for in_dim in stage_dims
        ])

    def _build_vig_feature_backbone(self, backbone: str) -> Tuple[nn.Module, int, List[int]]:
        """Build the selectable branch backbone and expose three spatial stages."""
        if backbone in RAW_ONLY_BACKBONES:
            raise ValueError(f"{backbone} does not support branch/fusion feature mode")
        if backbone == "vit":
            model = ViTBackbone(
                dim=self.vit_dim,
                depth=self.vit_depth,
                num_heads=self.vit_num_heads,
                num_classes=None,
            )
            return model, model.dim, [model.dim, model.dim, model.dim]
        if backbone in {"convnext_tiny", "vmamba_tiny"}:
            return build_pyramid_backbone(backbone)
        if backbone == "vig_b":
            return timm.create_model('vig_b', pretrained=False, num_classes=0), 640, [640, 640, 640]
        return timm.create_model('vig_s', pretrained=False, num_classes=0), 320, [320, 320, 320]

    def _build_vig_classifier(self, backbone: str, num_classes: int) -> nn.Module:
        if backbone in RAW_ONLY_BACKBONES:
            return build_raw_classifier(backbone, num_classes)
        if backbone == "vit":
            return ViTBackbone(
                dim=self.vit_dim,
                depth=self.vit_depth,
                num_heads=self.vit_num_heads,
                num_classes=num_classes,
            )
        if backbone in {"convnext_tiny", "vmamba_tiny"}:
            model, _, _ = build_pyramid_backbone(backbone, num_classes=num_classes)
            return model
        if backbone == "vig_b":
            return timm.create_model('vig_b', pretrained=False, num_classes=num_classes)
        return timm.create_model('vig_s', pretrained=False, num_classes=num_classes)

    @staticmethod
    def _build_resnet_feature_backbone(backbone: str) -> Tuple[nn.Module, int, List[int]]:
        if backbone == "50":
            model = models.resnet50(weights=None)
            stage_dims = [512, 1024, 2048]
            feat_dim = 2048
        else:
            model = models.resnet18(weights=None)
            stage_dims = [128, 256, 512]
            feat_dim = 512

        model.fc = nn.Identity()
        return model, feat_dim, stage_dims

    @staticmethod
    def _build_resnet_classifier(backbone: str, num_classes: int) -> nn.Module:
        if backbone == "50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _forward_resnet_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        stage_1 = self.resnet.layer2(x)
        stage_2 = self.resnet.layer3(stage_1)
        stage_3 = self.resnet.layer4(stage_2)
        return [stage_1, stage_2, stage_3]

    def _forward_resnet_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate feature vector for either raw ResNet variant."""
        r = self.resnet.resnet if isinstance(self.resnet, GloReResNet) else self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x = r.layer1(x)
        if isinstance(self.resnet, GloReResNet):
            x = self.resnet.glore2(r.layer2(x))
            x = self.resnet.glore3(r.layer3(x))
        else:
            x = r.layer2(x)
            x = r.layer3(x)
        x = r.layer4(x)
        return torch.flatten(r.avgpool(x), 1)

    def _forward_vig_raw(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """Compute raw ViG/ViT logits and features in one backbone pass."""
        features = self.vig.forward_features(x)
        head_logits = None
        if hasattr(self.vig, 'forward_head_logits'):
            head_logits = self.vig.forward_head_logits(features)
            logits = sum(head_logits)
        elif hasattr(self.vig, 'forward_head'):
            logits = self.vig.forward_head(features)
            features = self.vig.forward_head(features, pre_logits=True)
        elif hasattr(self.vig, 'prediction') and not isinstance(self.vig.prediction, nn.Identity):
            logits = self.vig.prediction(features.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        elif getattr(self.vig, 'head', None) is not None:
            logits = self.vig.head(features)
        else:
            logits = self.vig(x)
        return logits, features, head_logits

    def _project_stage_features(self, stage_maps: List[torch.Tensor], projectors: nn.ModuleList) -> List[torch.Tensor]:
        stage_features = []
        for stage_map, projector in zip(stage_maps, projectors):
            x = projector(stage_map)
            x = nn.functional.adaptive_avg_pool2d(x, self.graph_pool)
            x = x.flatten(2).transpose(1, 2).contiguous()
            stage_features.append(x)
        return stage_features

    @staticmethod
    def _global_feature(stage_map: torch.Tensor, projector: nn.Linear, norm: nn.LayerNorm) -> torch.Tensor:
        pooled = nn.functional.adaptive_avg_pool2d(stage_map, 1).flatten(1)
        return norm(projector(pooled))

    def forward(self, x) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
        stage_features: Dict[str, List[torch.Tensor]] = {}
        branch_features: Dict[str, torch.Tensor] = {}

        if self.mode in FUSION_MODES:
            vig_stage_maps = self.vig.forward_stage_features(x)
            res_stage_maps = self._forward_resnet_stages(x)

            stage_features = {
                'vig': self._project_stage_features(vig_stage_maps, self.vig_stage_projectors),
                'res': self._project_stage_features(res_stage_maps, self.res_stage_projectors),
            }

            feat_vig = self._global_feature(vig_stage_maps[-1], self.proj_vig, self.norm_vig)
            feat_res = self._global_feature(res_stage_maps[-1], self.proj_res, self.norm_res)
            branch_features['feat_vig'] = feat_vig
            branch_features['feat_res'] = feat_res

            if self.use_cross_attn:
                fused = self.fusion(feat_vig, feat_res)
            else:
                fused = torch.cat([feat_vig, feat_res], dim=1)
            feature_logits = self.classifier(fused)

            if self.mode == "prob_fusion":
                vig_logits = self.vig_classifier(feat_vig)
                res_logits = self.res_classifier(feat_res)
                probabilities = torch.stack([
                    F.softmax(vig_logits.float(), dim=1),
                    F.softmax(res_logits.float(), dim=1),
                    F.softmax(feature_logits.float(), dim=1),
                ]).mean(dim=0)
                logits = probabilities.clamp_min(1e-12).log()
                branch_features['prob_aux_logits'] = [vig_logits, res_logits, feature_logits]
            else:
                logits = feature_logits

        elif self.mode == "vig":
            vig_stage_maps = self.vig.forward_stage_features(x)
            feat_vig = self._global_feature(vig_stage_maps[-1], self.proj_vig, self.norm_vig)
            branch_features['feat_vig'] = feat_vig
            logits = self.classifier(feat_vig)

        elif self.mode == "resnet":
            res_stage_maps = self._forward_resnet_stages(x)
            feat_res = self._global_feature(res_stage_maps[-1], self.proj_res, self.norm_res)
            branch_features['feat_res'] = feat_res
            logits = self.classifier(feat_res)

        elif self.mode == "vig_raw":
            logits, feat_vig, head_logits = self._forward_vig_raw(x)
            if head_logits is not None:
                branch_features['aux_logits'] = head_logits
            if not self.training:
                branch_features['feat_vig'] = feat_vig

        elif self.mode == "resnet_raw":
            feat_res = self._forward_resnet_raw_features(x)
            r = self.resnet.resnet if isinstance(self.resnet, GloReResNet) else self.resnet
            logits = r.fc(feat_res)
            branch_features['feat_res'] = feat_res

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return logits, stage_features, branch_features


class MultiStageGraphDistance(nn.Module):
    def __init__(
            self,
            k: int = 8,
            eig_dim: int = 8,
            tau: float = 0.5,
            alpha: float = 1.0,
            beta: float = 0.2,
            stage_weights: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            graph_use: str = "both",
            spectral_method: str = "eigenvalue",
            kde_points: int = 64,
            kde_bandwidth: float = 0.0,
            kde_bandwidth_scale: float = 1.0,
            kde_eig_jitter: float = 1e-3,
    ):
        super().__init__()
        self.k = k
        self.eig_dim = eig_dim
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.stage_weights = stage_weights
        self.graph_use = graph_use  # "both", "adj-only", "spec-only"
        self.spectral_method = spectral_method  # "eigenvalue", "cdf", "kde-cdf"
        self.kde_points = kde_points
        self.kde_bandwidth = kde_bandwidth
        self.kde_bandwidth_scale = kde_bandwidth_scale
        self.kde_eig_jitter = kde_eig_jitter

    @staticmethod
    def _pairwise_distance(x: torch.Tensor) -> torch.Tensor:
        x_square = (x * x).sum(dim=-1, keepdim=True)
        dist = x_square + x_square.transpose(1, 2) - 2 * torch.bmm(x, x.transpose(1, 2))
        return dist.clamp_min(0.0)

    @staticmethod
    def _eye_mask(batch: int, nodes: int, device: torch.device) -> torch.Tensor:
        return torch.eye(nodes, device=device, dtype=torch.bool).unsqueeze(0).expand(batch, -1, -1)

    def _soft_adjacency(self, dist: torch.Tensor, hard: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute soft adjacency only within top-k neighbors.
        Symmetrize first, then mask with hard to guarantee same support.
        """
        batch, nodes, _ = dist.shape
        eye = self._eye_mask(batch, nodes, dist.device)

        # Gather distances only for top-k neighbors
        k = knn_idx.shape[-1]
        gathered_dist = torch.gather(dist, -1, knn_idx)  # [B, N, k]

        # Softmax over k neighbors
        logits = -gathered_dist / max(self.tau, 1e-6)
        soft_k = torch.softmax(logits, dim=-1)  # [B, N, k]

        # Scatter back to full adjacency matrix
        soft = torch.zeros_like(dist)
        soft.scatter_(-1, knn_idx, soft_k)

        # Symmetrize first (before masking for better numerical properties)
        soft = 0.5 * (soft + soft.transpose(1, 2))

        # Mask with hard to guarantee same support
        soft = soft * hard

        # Remove self-loops
        soft = soft.masked_fill(eye, 0.0)

        return soft

    def _hard_knn_adjacency(self, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return hard binary adjacency and KNN indices for soft adjacency computation."""
        batch, nodes, _ = dist.shape
        eye = self._eye_mask(batch, nodes, dist.device)
        k = min(self.k, max(nodes - 1, 1))
        knn_dist = dist.masked_fill(eye, float('inf'))
        knn_idx = torch.topk(knn_dist, k=k, dim=-1, largest=False).indices

        adj = torch.zeros_like(dist)
        adj.scatter_(-1, knn_idx, 1.0)
        adj = torch.maximum(adj, adj.transpose(1, 2))  # Symmetrize
        adj = adj.masked_fill(eye, 0.0)

        return adj, knn_idx

    def _ste_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator:
        - Forward uses hard binary adjacency (0/1)
        - Backward uses soft adjacency gradient within KNN neighbors
        - L2 normalize features before distance computation for stable distance
        """
        # L2 normalize features for stable pairwise distance
        x = nn.functional.normalize(x, p=2, dim=-1)

        dist = self._pairwise_distance(x)
        hard, knn_idx = self._hard_knn_adjacency(dist)
        soft = self._soft_adjacency(dist, hard, knn_idx)

        # STE: forward = hard, backward = soft gradient
        return hard + (soft - soft.detach())

    @staticmethod
    def _normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = adj.shape
        eye = torch.eye(nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(batch, -1, -1)

        # Ensure adjacency is symmetric
        adj = 0.5 * (adj + adj.transpose(1, 2))
        adj = adj * (1.0 - eye)

        degree = adj.sum(dim=-1)
        deg_inv_sqrt = (degree + 1e-6).rsqrt()
        norm_adj = adj * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

        laplacian = eye - norm_adj

        # Ensure Laplacian is symmetric for numerical stability
        laplacian = 0.5 * (laplacian + laplacian.transpose(1, 2))

        return laplacian

    def _spectral_distance(self, adj_a: torch.Tensor, adj_b: torch.Tensor) -> torch.Tensor:
        if self.spectral_method == "cdf":
            with torch.no_grad():
                cdf_distance = self._spectral_cdf_distance(adj_a, adj_b)
            proxy = self._spectral_eigenvalue_distance(adj_a, adj_b)
            return cdf_distance.detach() + proxy - proxy.detach()
        if self.spectral_method == "kde-cdf":
            kde_distance = self._spectral_kde_cdf_distance(adj_a, adj_b)
            if not torch.isfinite(kde_distance).all().item():
                raise RuntimeError("kde-cdf spectral distance produced a non-finite value")
            return kde_distance

        return self._spectral_eigenvalue_distance(adj_a, adj_b)

    def _spectral_eigenvalue_distance(self, adj_a: torch.Tensor, adj_b: torch.Tensor) -> torch.Tensor:
        lap_a = self._normalized_laplacian(adj_a.float())
        lap_b = self._normalized_laplacian(adj_b.float())

        eig_a = torch.linalg.eigvalsh(lap_a)
        eig_b = torch.linalg.eigvalsh(lap_b)

        eig_dim = min(self.eig_dim, eig_a.shape[-1], eig_b.shape[-1])
        return ((eig_a[:, :eig_dim] - eig_b[:, :eig_dim]) ** 2).sum(dim=-1).mean()

    @staticmethod
    def _unnormalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = adj.shape
        eye = torch.eye(nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(batch, -1, -1)

        adj = 0.5 * (adj + adj.transpose(1, 2))
        adj = adj * (1.0 - eye)
        degree = adj.sum(dim=-1)

        laplacian = torch.diag_embed(degree) - adj
        return 0.5 * (laplacian + laplacian.transpose(1, 2))

    @staticmethod
    def _normalize_eigenvector_distribution(vec: torch.Tensor) -> torch.Tensor:
        max_abs_idx = vec.abs().argmax(dim=-1, keepdim=True)
        signs = torch.gather(vec, -1, max_abs_idx).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        vec = vec * signs

        min_val = vec.min(dim=-1, keepdim=True).values
        max_val = vec.max(dim=-1, keepdim=True).values
        return (vec - min_val) / (max_val - min_val + 1e-6)

    def _nontrivial_eigenvector_distributions(
            self,
            adj_a: torch.Tensor,
            adj_b: torch.Tensor,
            eig_jitter: float = 0.0,
            use_float64: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.float64 if use_float64 else torch.float32
        lap_a = self._unnormalized_laplacian(adj_a.to(dtype))
        lap_b = self._unnormalized_laplacian(adj_b.to(dtype))

        if eig_jitter > 0.0:
            nodes = lap_a.shape[-1]
            jitter = torch.linspace(-0.5, 0.5, nodes, dtype=lap_a.dtype, device=lap_a.device)
            jitter = torch.diag_embed(jitter).unsqueeze(0)
            lap_a = lap_a + eig_jitter * jitter
            lap_b = lap_b + eig_jitter * jitter

        _, eigvec_a = torch.linalg.eigh(lap_a)
        _, eigvec_b = torch.linalg.eigh(lap_b)

        max_vecs = min(eigvec_a.shape[-1], eigvec_b.shape[-1]) - 1
        eig_dim = min(self.eig_dim, max_vecs)
        if eig_dim < 1:
            empty_shape = (adj_a.shape[0], 0, adj_a.shape[1])
            return adj_a.new_empty(empty_shape), adj_b.new_empty(empty_shape)

        vec_a = eigvec_a[:, :, 1:eig_dim + 1].transpose(1, 2)
        vec_b = eigvec_b[:, :, 1:eig_dim + 1].transpose(1, 2)
        return (
            self._normalize_eigenvector_distribution(vec_a),
            self._normalize_eigenvector_distribution(vec_b),
        )

    def _spectral_cdf_distance(self, adj_a: torch.Tensor, adj_b: torch.Tensor) -> torch.Tensor:
        """
        Shimada et al. graph distance: average L1 distance between empirical
        CDFs of Laplacian eigenvector-element distributions, excluding v1.
        """
        vec_a, vec_b = self._nontrivial_eigenvector_distributions(adj_a, adj_b)
        if vec_a.shape[1] == 0:
            return adj_a.new_tensor(0.0)

        # For equally sized empirical samples, integral |CDF_a - CDF_b| dy
        # equals the mean absolute difference of sorted samples.
        sorted_a = vec_a.sort(dim=-1).values
        sorted_b = vec_b.sort(dim=-1).values
        return (sorted_a - sorted_b).abs().mean(dim=-1).mean()

    def _kde_bandwidth(self, nodes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.kde_bandwidth > 0.0:
            bandwidth = self.kde_bandwidth
        else:
            bandwidth = self.kde_bandwidth_scale * (float(nodes) ** (-1.0 / 3.0))
        return torch.tensor(max(bandwidth, 1e-3), dtype=dtype, device=device)

    def _spectral_kde_cdf_distance(self, adj_a: torch.Tensor, adj_b: torch.Tensor) -> torch.Tensor:
        """
        Smooth KDE-CDF graph distance. Unlike cdf mode, this path keeps the
        KDE-CDF computation in the autograd graph for ablation experiments.
        """
        vec_a, vec_b = self._nontrivial_eigenvector_distributions(
            adj_a,
            adj_b,
            eig_jitter=self.kde_eig_jitter,
            use_float64=True,
        )
        if vec_a.shape[1] == 0:
            return adj_a.new_tensor(0.0)

        points = max(int(self.kde_points), 2)
        grid = torch.linspace(0.0, 1.0, points, dtype=vec_a.dtype, device=vec_a.device)
        bandwidth = self._kde_bandwidth(vec_a.shape[-1], vec_a.dtype, vec_a.device)
        denom = bandwidth * math.sqrt(2.0)

        cdf_a = 0.5 * (1.0 + torch.erf((grid.view(1, 1, 1, -1) - vec_a.unsqueeze(-1)) / denom))
        cdf_b = 0.5 * (1.0 + torch.erf((grid.view(1, 1, 1, -1) - vec_b.unsqueeze(-1)) / denom))
        cdf_a = cdf_a.mean(dim=2)
        cdf_b = cdf_b.mean(dim=2)

        diff = cdf_a - cdf_b
        smooth_abs = torch.sqrt(diff * diff + 1e-6) - 1e-3
        return smooth_abs.mean().to(adj_a.dtype)

    @staticmethod
    def _enabled_stage_indices(epoch: int) -> List[int]:
        if epoch <= 10:
            return [2]
        if epoch <= 20:
            return [1, 2]
        return [0, 1, 2]

    def forward(
            self,
            stage_features: Dict[str, List[torch.Tensor]],
            epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        res_stages = stage_features.get('res', [])
        vig_stages = stage_features.get('vig', [])

        # Fixed: Check for exact length requirement
        if len(res_stages) < 3 or len(vig_stages) < 3:
            raise ValueError(
                f"Expected at least 3 stages from both branches, "
                f"got {len(res_stages)} res stages and {len(vig_stages)} vig stages"
            )

        # Initialize with proper device and dtype
        device = res_stages[0].device
        dtype = res_stages[0].dtype
        zero = torch.tensor(0.0, device=device, dtype=dtype)

        total = zero.clone()
        metrics: Dict[str, torch.Tensor] = {}
        enabled = set(self._enabled_stage_indices(epoch))

        for stage_idx in range(3):
            adj_key = f'adj_s{stage_idx + 1}'
            spec_key = f'spec_s{stage_idx + 1}'

            # Initialize metrics (no need for detach on zero)
            metrics[adj_key] = zero.clone()
            metrics[spec_key] = zero.clone()

            if stage_idx not in enabled:
                continue

            # Compute adjacency matrices with STE
            adj_res = self._ste_adjacency(res_stages[stage_idx])
            adj_vig = self._ste_adjacency(vig_stages[stage_idx])

            # Conditional computation based on graph_use
            if self.graph_use in ["both", "adj-only"]:
                # Adjacency distance
                loss_adj = ((adj_res - adj_vig) ** 2).mean()
            else:
                # spec-only: no adjacency loss
                loss_adj = zero.clone()

            # Spectral distance (with error handling)
            if self.graph_use in ["both", "spec-only"] and self.beta > 0.0 and self.eig_dim > 0:
                try:
                    loss_spec = self._spectral_distance(adj_res, adj_vig)
                except RuntimeError as e:
                    # Log warning if needed
                    loss_spec = zero.clone()
            else:
                loss_spec = zero.clone()

            # Store metrics
            metrics[adj_key] = loss_adj.detach()
            metrics[spec_key] = loss_spec.detach()

            # Accumulate weighted loss
            if self.graph_use == "spec-only":
                # spec-only: alpha = 0
                stage_loss = self.beta * loss_spec
            elif self.graph_use == "adj-only":
                # adj-only: beta = 0
                stage_loss = self.alpha * loss_adj
            else:
                # both: alpha L_adj + beta L_spec
                stage_loss = self.alpha * loss_adj + self.beta * loss_spec
            total = total + self.stage_weights[stage_idx] * stage_loss

        # Store final metrics
        metrics['graph'] = total.detach()
        metrics['enabled_stage_count'] = torch.tensor(
            float(len(enabled)),
            device=device,
            dtype=dtype
        )

        return total, metrics


class FusionLoss(nn.Module):
    def __init__(
        self,
        mode: str = "fusion",
        graph_loss_weight: float = 0.1,
        graph_alpha: float = 1.0,
        graph_beta: float = 0.2,
        graph_k: int = 8,
        graph_eig_dim: int = 8,
        graph_tau: float = 0.5,
        graph_stage_weights: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        graph_use: str = "both",
        graph_spectral_method: str = "eigenvalue",
        graph_kde_points: int = 64,
        graph_kde_bandwidth: float = 0.0,
        graph_kde_bandwidth_scale: float = 1.0,
        graph_kde_eig_jitter: float = 1e-3,
        graph_distance_mode: str = "push-apart",
        align_weight: float = 0.0,
        graph_anneal: bool = False,
        anneal_start: float = 0.5,
        max_epochs: int = 300,
        prob_aux_weight: float = 0.3,
    ):
        super().__init__()
        if graph_distance_mode not in ("push-apart", "pull-close", "off"):
            raise ValueError(f"Unsupported graph_distance_mode: {graph_distance_mode}")
        self.ce = nn.CrossEntropyLoss()
        self.mode = mode
        self.graph_loss_weight = graph_loss_weight
        self.graph_distance_mode = graph_distance_mode
        self.align_weight = align_weight
        self.graph_anneal = graph_anneal
        self.anneal_start = anneal_start
        self.max_epochs = max_epochs
        self.prob_aux_weight = prob_aux_weight
        self.graph_regularizer = None
        self.use_graph_loss = (
            self.mode in FUSION_MODES
            and self.graph_loss_weight > 0.0
            and self.graph_distance_mode != "off"
        )
        if self.use_graph_loss:
            self.graph_regularizer = MultiStageGraphDistance(
                k=graph_k,
                eig_dim=graph_eig_dim,
                tau=graph_tau,
                alpha=graph_alpha,
                beta=graph_beta,
                stage_weights=graph_stage_weights,
                graph_use=graph_use,
                spectral_method=graph_spectral_method,
                kde_points=graph_kde_points,
                kde_bandwidth=graph_kde_bandwidth,
                kde_bandwidth_scale=graph_kde_bandwidth_scale,
                kde_eig_jitter=graph_kde_eig_jitter,
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        stage_features: Dict[str, List[torch.Tensor]],
        epoch: int,
        branch_features: Dict[str, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux_logits = branch_features.get('aux_logits') if branch_features is not None else None
        prob_aux_logits = branch_features.get('prob_aux_logits') if branch_features is not None else None
        if self.mode == "prob_fusion" and isinstance(prob_aux_logits, (list, tuple)) and prob_aux_logits:
            loss_main = self.ce(logits, targets)
            loss_aux = sum(self.ce(head, targets) for head in prob_aux_logits) / len(prob_aux_logits)
            loss_cls = loss_main + self.prob_aux_weight * loss_aux
        elif isinstance(aux_logits, (list, tuple)) and aux_logits:
            head_losses = [self.ce(head, targets) for head in aux_logits]
            loss_cls = sum(head_losses) / len(head_losses)
        else:
            loss_cls = self.ce(logits, targets)
        metrics: Dict[str, torch.Tensor] = {'ce': loss_cls.detach()}
        if self.mode == "prob_fusion" and isinstance(prob_aux_logits, (list, tuple)) and prob_aux_logits:
            metrics['ce_main'] = loss_main.detach()
            metrics['ce_aux'] = loss_aux.detach()
            metrics['prob_aux_weight'] = torch.tensor(
                self.prob_aux_weight,
                device=logits.device,
                dtype=logits.dtype,
            )

        total = loss_cls

        if self.use_graph_loss and self.graph_regularizer is not None:
            graph_loss, graph_metrics = self.graph_regularizer(stage_features, epoch)
            metrics.update(graph_metrics)
            lambda_g = self.graph_loss_weight
            if self.graph_anneal:
                progress = epoch / self.max_epochs
                p = max(0, (progress - self.anneal_start) / (1.0 - self.anneal_start + 1e-8))
                lambda_g = self.graph_loss_weight * 0.5 * (1 + math.cos(math.pi * p))
            metrics['lambda_g'] = torch.tensor(lambda_g)
            if self.graph_distance_mode == "push-apart":
                total = total - lambda_g * graph_loss
            elif self.graph_distance_mode == "pull-close":
                total = total + lambda_g * graph_loss

        if self.align_weight > 0.0 and branch_features is not None:
            feat_vig = branch_features.get('feat_vig')
            feat_res = branch_features.get('feat_res')
            if feat_vig is not None and feat_res is not None:
                cos_sim = F.cosine_similarity(feat_vig, feat_res, dim=1)
                loss_align = (1 - cos_sim).mean()
                metrics['align'] = loss_align.detach()
                total = total + self.align_weight * loss_align

        metrics['total'] = total.detach()
        return total, metrics


def build_classification_loaders(
    data_dir: str,
    batch_size: int,
    workers: int = 4,
    dataset: str = 'cifar100',
    val_ratio: float = 0.1,
    split_seed: int = 42,
    color_jitter: float = 0.0,
    random_rotation: float = 0.0,
    random_grayscale: float = 0.0,
    random_erasing: float = 0.0,
    rand_augment_n: int = 0,
    rand_augment_m: int = 0,
):
    """
    Build train/validation/test loaders without using the official test split
    for checkpoint selection.

    Augmentation is purely optional: every flag defaults to off, in which case
    the train transform is identical to the original pipeline
    (RandomResizedCrop + RandomHorizontalFlip + ToTensor + Normalize). Extra
    augmentations are appended in-place and never touch the validation
    transform, so ViG/ResNet graph characteristics are unaffected -- both
    branches always observe the same augmented image.
    """
    normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    train_tf_list = [
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if random_rotation > 0.0:
        train_tf_list.append(transforms.RandomRotation(random_rotation))
    if color_jitter > 0.0:
        train_tf_list.append(transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=min(color_jitter * 0.1, 0.5),
        ))
    if random_grayscale > 0.0:
        train_tf_list.append(transforms.RandomGrayscale(p=random_grayscale))
    if rand_augment_n > 0:
        train_tf_list.append(
            transforms.RandAugment(num_ops=rand_augment_n, magnitude=rand_augment_m)
        )
    train_tf_list.append(transforms.ToTensor())
    train_tf_list.append(normalize)
    if random_erasing > 0.0:
        # RandomErasing operates on tensors, so it must follow ToTensor/Normalize.
        train_tf_list.append(transforms.RandomErasing(p=random_erasing))

    train_tf = transforms.Compose(train_tf_list)
    if dataset == 'tiny-imagenet-200':
        val_tf_list = [transforms.Resize((224, 224))]
    else:
        val_tf_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    val_tf = transforms.Compose(val_tf_list + [transforms.ToTensor(), normalize])

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f'val_ratio must be between 0 and 1, got {val_ratio}')

    def stratified_split_indices(targets):
        targets = np.asarray(targets)
        rng = np.random.default_rng(split_seed)
        train_indices = []
        val_indices = []
        for class_id in np.unique(targets):
            class_indices = np.flatnonzero(targets == class_id)
            rng.shuffle(class_indices)
            val_count = int(round(len(class_indices) * val_ratio))
            val_count = max(1, min(len(class_indices) - 1, val_count))
            val_indices.extend(class_indices[:val_count].tolist())
            train_indices.extend(class_indices[val_count:].tolist())
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        return train_indices, val_indices

    if dataset == 'tiny-imagenet-200':
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            raise FileNotFoundError(
                f'Tiny ImageNet-200 needs train/ and val/ class folders under {data_dir}. '
                'Run prepare_tiny_imagenet.py first.'
            )
        full_train_set = datasets.ImageFolder(train_dir, transform=train_tf)
        full_val_set = datasets.ImageFolder(train_dir, transform=val_tf)
        test_set = datasets.ImageFolder(val_dir, transform=val_tf)
        if len(full_train_set.classes) != 200 or full_train_set.classes != test_set.classes:
            raise ValueError('Tiny ImageNet-200 train/ and val/ must contain the same 200 class folders')
        train_indices, val_indices = stratified_split_indices(full_train_set.targets)
    else:
        full_train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
        full_val_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=val_tf)
        test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=val_tf)
        train_indices, val_indices = stratified_split_indices(full_train_set.targets)

    train_set = Subset(full_train_set, train_indices)
    val_set = Subset(full_val_set, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100.0


def _update_metric_sums(metric_sums: Dict[str, float], parts: Dict[str, torch.Tensor], batch_size: int):
    for key, value in parts.items():
        if isinstance(value, torch.Tensor):
            value = value.item()
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size


def _finalize_metric_sums(metric_sums: Dict[str, float], dataset_size: int) -> Dict[str, float]:
    if dataset_size == 0:
        return {}
    return {key: value / dataset_size for key, value in metric_sums.items()}


def train_one_epoch(model, loader, optim, loss_fn, device, epoch: int, amp: bool = True, scaler=None):
    model.train()
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    running, running_acc = 0.0, 0.0
    metric_sums: Dict[str, float] = {}

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits, stage_features, branch_features = model(images)
            loss, parts = loss_fn(logits, labels, stage_features, epoch, branch_features)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        batch_size = images.size(0)
        running += loss.item() * batch_size
        running_acc += accuracy(logits.detach(), labels) * batch_size
        _update_metric_sums(metric_sums, parts, batch_size)

    n = len(loader.dataset)
    return running / n, running_acc / n, _finalize_metric_sums(metric_sums, n)


def _atomic_torch_save(payload: Dict[str, object], path: str):
    """Write the best-model checkpoint atomically."""
    temp_path = f'{path}.tmp'
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _cpu_state(value):
    """Copy tensors off the GPU without altering the live training state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    return value


def _save_training_checkpoint(payload, path):
    """Keep the previous checkpoint until the new file has been flushed to disk."""
    temp_path = path + '.tmp'
    with open(temp_path, 'wb') as handle:
        torch.save(_cpu_state(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    if os.path.isfile(path):
        os.replace(path, path + '.prev')
    os.replace(temp_path, path)


def _load_training_checkpoint(path):
    required = {'model', 'optimizer', 'scheduler', 'scaler', 'history', 'rng',
                'epoch', 'args', 'model_config', 'best_acc', 'best_epoch', 'best_filename'}
    errors = []
    for candidate in (path, path + '.prev'):
        try:
            checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)
            if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
                raise ValueError('not a complete training checkpoint')
            return checkpoint, candidate
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            errors.append(f'{candidate}: {exc}')
    raise RuntimeError('No readable training checkpoint or backup:\n' + '\n'.join(errors))


def evaluate(model, loader, loss_fn, device, epoch: int, amp: bool = True):
    model.eval()
    running, running_acc = 0.0, 0.0
    metric_sums: Dict[str, float] = {}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits, stage_features, branch_features = model(images)
                loss, parts = loss_fn(logits, labels, stage_features, epoch, branch_features)

            batch_size = images.size(0)
            running += loss.item() * batch_size
            running_acc += accuracy(logits, labels) * batch_size
            _update_metric_sums(metric_sums, parts, batch_size)

    n = len(loader.dataset)
    return running / n, running_acc / n, _finalize_metric_sums(metric_sums, n)


def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, learning_rates, args):
    epochs = range(1, len(train_losses) + 1)

    with PdfPages(f'training_curves_{args.exp_name}_{args.dataset.replace("-", "_")}.pdf') as pdf:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2, marker='^', markersize=4)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        ax4.axis('off')
        final_train_acc = train_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        best_val_acc = max(val_accuracies)
        best_val_epoch = val_accuracies.index(best_val_acc) + 1
        min_train_loss = min(train_losses)
        min_val_loss = min(val_losses)

        stats_text = f"""Training Statistics Summary:

Final Training Accuracy: {final_train_acc:.2f}%
Final Validation Accuracy: {final_val_acc:.2f}%
Best Validation Accuracy: {best_val_acc:.2f}%
Best Validation Epoch: {best_val_epoch}

Minimum Training Loss: {min_train_loss:.4f}
Minimum Validation Loss: {min_val_loss:.4f}

Model Configuration:
• Mode: {args.mode}
• ResNet Backbone: {args.resnet_backbone}
• Total Epochs: {args.epochs}
• Initial Learning Rate: {args.lr}
• Final Learning Rate: {learning_rates[-1]:.2e}
• Weight Decay: {args.weight_decay}
• Batch Size: {args.batch_size}"""

        if args.vig_backbone == 'vit':
            stats_text += f"""
• ViT Config: dim={args.vit_dim} depth={args.vit_depth} heads={args.vit_num_heads}"""

        if getattr(args, 'use_glore', False):
            stats_text += f"""
• GloRe: enabled (mid ratio {args.glore_mid_ratio})"""

        if args.mode in FUSION_MODES:
            stats_text += f"""
• Graph Loss Weight: {args.graph_loss_weight}
• Graph Alpha: {args.graph_alpha}
• Graph Beta: {args.graph_beta}
• Graph K: {args.graph_k}
• Graph Eig Dim: {args.graph_eig_dim}
• Graph Spectral Method: {args.graph_spectral_method}
• Graph Distance Mode: {args.graph_distance_mode}
• Graph KDE Points: {args.graph_kde_points}
• Graph KDE Bandwidth: {args.graph_kde_bandwidth}
• Graph KDE Bandwidth Scale: {args.graph_kde_bandwidth_scale}
• Graph KDE Eig Jitter: {args.graph_kde_eig_jitter}
• Stage Weights: {args.stage_weights}
• Graph Use: {args.graph_use}"""

        if args.mode == "prob_fusion":
            stats_text += f"""
• Probability Fusion: equal-weight mean of ViG, ResNet, and fused-feature probabilities
• Probability Auxiliary Weight: {args.prob_aux_weight}"""

        ax4.text(
            0.05,
            0.95,
            stats_text,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            transform=ax4.transAxes,
        )

        plt.suptitle(f'Training Results - {args.mode.upper()} Mode', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=3, alpha=0.8)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=3, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Loss Curves - {args.mode.upper()}', fontsize=14, fontweight='bold')

        ax2.plot(epochs, train_accuracies, 'b-', label='Training Acc', linewidth=3, marker='o', markersize=5, alpha=0.8)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Acc', linewidth=3, marker='s', markersize=5, alpha=0.8)
        best_idx = val_accuracies.index(best_val_acc)
        ax2.plot(best_idx + 1, best_val_acc, 'g*', markersize=15, label=f'Best Val Acc: {best_val_acc:.2f}%')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Accuracy Curves - {args.mode.upper()}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()


def _format_epoch_metrics(metrics: Dict[str, float], mode: str) -> str:
    message = f"ce={metrics.get('ce', 0.0):.4f}"
    if mode == "prob_fusion":
        message += (
            f" main={metrics.get('ce_main', 0.0):.4f}"
            f" aux={metrics.get('ce_aux', 0.0):.4f}"
            f" aux_w={metrics.get('prob_aux_weight', 0.0):.2f}"
        )
    if mode in FUSION_MODES:
        enabled_count = int(round(metrics.get('enabled_stage_count', 0.0)))
        message += (
            f" graph={metrics.get('graph', 0.0):.4f}"
            f" adj3={metrics.get('adj_s3', 0.0):.4f}"
            f" spec3={metrics.get('spec_s3', 0.0):.4f}"
            f" stages={enabled_count}"
        )
        lambda_g = metrics.get('lambda_g', None)
        if lambda_g is not None:
            message += f" λ_g={lambda_g:.5f}"
    return message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data',
                        help='CIFAR-100 cache root or prepared Tiny ImageNet-200 root')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'tiny-imagenet-200'],
                        help='image classification dataset (default: cifar100)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='stratified fraction of the training split used for validation (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None,
                        help='restore a full last checkpoint (training settings are restored automatically)')
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help='save full training state every N epochs and at the end (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--aug-color-jitter', type=float, default=0.0,
                        help='ColorJitter strength for brightness/contrast/saturation (0=disabled, e.g. 0.4)')
    parser.add_argument('--aug-random-rotation', type=float, default=0.0,
                        help='max random rotation in degrees (0=disabled, e.g. 15)')
    parser.add_argument('--aug-random-grayscale', type=float, default=0.0,
                        help='random grayscale probability (0=disabled, e.g. 0.2)')
    parser.add_argument('--aug-random-erasing', type=float, default=0.0,
                        help='random erasing probability, applied after normalization (0=disabled, e.g. 0.25)')
    parser.add_argument('--aug-rand-augment-n', type=int, default=0,
                        help='RandAugment number of ops (0=disabled, e.g. 2)')
    parser.add_argument('--aug-rand-augment-m', type=int, default=0,
                        help='RandAugment magnitude in [0, 30] (0=disabled; set together with --aug-rand-augment-n, e.g. 10)')
    parser.add_argument('--freeze-resnet', action='store_true')
    parser.add_argument('--freeze-vig', action='store_true')
    parser.add_argument('--mode', type=str, default='fusion',
                        choices=['fusion', 'prob_fusion', 'vig', 'resnet', 'vig_raw', 'resnet_raw'])
    parser.add_argument('--use-cross-attention', action='store_true', help='use cross attention for feature fusion')
    parser.add_argument('--prob-aux-weight', type=float, default=0.3,
                        help='weight of the mean three-head auxiliary CE in prob_fusion mode (default: 0.3)')
    parser.add_argument('--resnet-backbone', type=str, default='18', choices=['18', '50'])
    parser.add_argument('--vig-backbone', type=str, default='vig_s',
                        choices=['vig_s', 'vig_b', 'vit', 'convnext_tiny', 'vmamba_tiny',
                                 'swin_tiny', 'pvt_v2_b2', 'conformer_tiny'],
                        help='selectable backbone (Swin-T, PVTv2-B2, and Conformer-Ti are vig_raw-only baselines)')
    parser.add_argument('--use-glore', action='store_true',
                        help='insert GloRe reasoning units into the ResNet backbone (resnet_raw mode only)')
    parser.add_argument('--glore-mid-ratio', type=float, default=0.5,
                        help='num_mid of each GloRe unit as a fraction of the stage channels (default: 0.5)')
    parser.add_argument('--vit-dim', type=int, default=384,
                        help='embedding dim of the ViT ablation backbone (default: 384, ~vig_s param budget)')
    parser.add_argument('--vit-depth', type=int, default=12,
                        help='number of transformer blocks of the ViT ablation backbone (default: 12)')
    parser.add_argument('--vit-num-heads', type=int, default=6,
                        help='number of attention heads of the ViT ablation backbone (default: 6)')
    parser.add_argument('--graph-k', type=int, default=8)
    parser.add_argument('--graph-loss-weight', type=float, default=0.1)
    parser.add_argument('--graph-alpha', type=float, default=1.0)
    parser.add_argument('--graph-beta', type=float, default=0.2)
    parser.add_argument('--graph-eig-dim', type=int, default=8)
    parser.add_argument('--graph-tau', type=float, default=0.5)
    parser.add_argument('--graph-spectral-method', type=str, default='eigenvalue', choices=['eigenvalue', 'cdf', 'kde-cdf'],
                        help='spectral distance backend: eigenvalue=old Laplacian eigenvalue distance, cdf=Shimada et al. CDF with proxy gradient, kde-cdf=smooth differentiable CDF')
    parser.add_argument('--graph-kde-points', type=int, default=64,
                        help='number of grid points for kde-cdf numerical integration')
    parser.add_argument('--graph-kde-bandwidth', type=float, default=0.0,
                        help='fixed KDE bandwidth for kde-cdf (<=0 uses adaptive bandwidth)')
    parser.add_argument('--graph-kde-bandwidth-scale', type=float, default=1.0,
                        help='scale for adaptive KDE bandwidth h=scale*n^(-1/3)')
    parser.add_argument('--graph-kde-eig-jitter', type=float, default=1e-3,
                        help='deterministic diagonal jitter added before kde-cdf eigh to reduce repeated-eigenvalue gradient instability')
    parser.add_argument('--graph-distance-mode', type=str, default='push-apart',
                        choices=['push-apart', 'pull-close', 'off'],
                        help='graph distance loss mode: push-apart keeps CE - lambda*graph_loss, pull-close uses CE + lambda*graph_loss, off disables the graph distance term')
    parser.add_argument('--stage-weights', type=str, default='last', choices=['last', 'three'],
                        help='stage weights: last=only last stage (0, 0, 1), three=three stages (0.2, 0.3, 0.5)')
    parser.add_argument('--graph-use', type=str, default='both', choices=['both', 'adj-only', 'spec-only'],
                        help='which loss terms to use: both=alpha*adj + beta*spec, adj-only=only adj, spec-only=only spec')
    parser.add_argument('--align-weight', type=float, default=0.0,
                        help='weight for cosine alignment loss between branch features (0=disabled)')
    parser.add_argument('--graph-anneal', action='store_true',
                        help='cosine anneal graph loss weight to 0 after anneal-start')
    parser.add_argument('--anneal-start', type=float, default=0.5,
                        help='fraction of training after which graph loss annealing begins (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    resume_checkpoint = None
    if args.checkpoint_every < 1:
        parser.error('--checkpoint-every must be at least 1')
    if args.resume:
        resume_checkpoint, loaded_path = _load_training_checkpoint(os.path.abspath(args.resume))
        # Runtime paths/workers and save frequency may change; training settings must not.
        for key, value in resume_checkpoint['args'].items():
            if key not in {'data', 'workers', 'resume', 'checkpoint_every'} and hasattr(args, key):
                setattr(args, key, value)
        print(f'Resuming from {loaded_path}; restored original training settings, total epochs={args.epochs}')
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')

    if args.use_glore and args.mode != "resnet_raw":
        parser.error("--use-glore is only supported in resnet_raw mode")
    if args.vig_backbone in RAW_ONLY_BACKBONES and args.mode != "vig_raw":
        parser.error(
            f"--vig-backbone {args.vig_backbone} is a single-model comparison baseline "
            "and requires --mode vig_raw"
        )
    if args.vig_backbone == "vit":
        if args.vit_dim <= 0:
            parser.error(f"--vit-dim must be positive, got {args.vit_dim}")
        if args.vit_num_heads <= 0:
            parser.error(f"--vit-num-heads must be positive, got {args.vit_num_heads}")
        if args.vit_dim % args.vit_num_heads != 0:
            parser.error(
                f"--vit-dim ({args.vit_dim}) must be divisible by "
                f"--vit-num-heads ({args.vit_num_heads})"
            )
        if args.vit_depth <= 0:
            parser.error(f"--vit-depth must be positive, got {args.vit_depth}")
        if args.mode in FUSION_MODES and args.vit_depth < 3:
            parser.error("fusion with --vig-backbone vit requires --vit-depth >= 3")
    if args.prob_aux_weight < 0.0:
        parser.error(f"--prob-aux-weight must be non-negative, got {args.prob_aux_weight}")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error(f"--val-ratio must be between 0 and 1, got {args.val_ratio}")

    # Parse stage weights
    if args.stage_weights == 'last':
        graph_stage_weights = (0.0, 0.0, 1.0)
    else:  # 'three'
        graph_stage_weights = (0.2, 0.3, 0.5)

    args.exp_name = args.mode
    if args.vig_backbone != 'vig_s':
        args.exp_name = f"{args.exp_name}_{args.vig_backbone}"
    if args.use_glore:
        args.exp_name = f"{args.exp_name}_glore"
    if args.mode in FUSION_MODES and args.graph_distance_mode != "push-apart":
        args.exp_name = f"{args.exp_name}_graph_{args.graph_distance_mode}"

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Validation Split: {args.val_ratio:.1%} of the training data (stratified, seed={args.seed})")
    print(f"Mode: {args.mode}")
    print(f"ResNet Backbone: {args.resnet_backbone}")
    print(f"Branch Backbone: {args.vig_backbone}")
    if args.mode == "prob_fusion":
        print("Probability Fusion Weights: ViG=1/3 ResNet=1/3 Feature=1/3")
        print(f"Probability Auxiliary Weight: {args.prob_aux_weight}")
    if args.vig_backbone == 'vit':
        print(f"ViT Config: dim={args.vit_dim} depth={args.vit_depth} heads={args.vit_num_heads}")
    print(f"Use GloRe: {args.use_glore}")
    if args.use_glore:
        print(f"GloRe Mid Ratio: {args.glore_mid_ratio}")
    print(f"Graph K: {args.graph_k}")
    print(f"Graph Loss Weight: {args.graph_loss_weight}")
    print(f"Graph Spectral Method: {args.graph_spectral_method}")
    print(f"Graph Distance Mode: {args.graph_distance_mode}")
    print(f"Graph KDE Points: {args.graph_kde_points}")
    print(f"Graph KDE Bandwidth: {args.graph_kde_bandwidth}")
    print(f"Graph KDE Bandwidth Scale: {args.graph_kde_bandwidth_scale}")
    print(f"Graph KDE Eig Jitter: {args.graph_kde_eig_jitter}")
    print(f"Align Weight: {args.align_weight}")
    print(f"Graph Anneal: {args.graph_anneal}")
    if args.graph_anneal:
        print(f"Anneal Start: {args.anneal_start}")

    active_augs = []
    if args.aug_color_jitter > 0.0:
        active_augs.append(f"color_jitter={args.aug_color_jitter}")
    if args.aug_random_rotation > 0.0:
        active_augs.append(f"rotation={args.aug_random_rotation}")
    if args.aug_random_grayscale > 0.0:
        active_augs.append(f"grayscale={args.aug_random_grayscale}")
    if args.aug_random_erasing > 0.0:
        active_augs.append(f"erasing={args.aug_random_erasing}")
    if args.aug_rand_augment_n > 0:
        active_augs.append(f"randaugment(n={args.aug_rand_augment_n},m={args.aug_rand_augment_m})")
    print(f"Data Augmentation: {', '.join(active_augs) if active_augs else 'none (baseline)'}")

    train_loader, val_loader, test_loader = build_classification_loaders(
        args.data,
        args.batch_size,
        args.workers,
        dataset=args.dataset,
        val_ratio=args.val_ratio,
        split_seed=args.seed,
        color_jitter=args.aug_color_jitter,
        random_rotation=args.aug_random_rotation,
        random_grayscale=args.aug_random_grayscale,
        random_erasing=args.aug_random_erasing,
        rand_augment_n=args.aug_rand_augment_n,
        rand_augment_m=args.aug_rand_augment_m,
    )
    print(
        f"Dataset Split Sizes: train={len(train_loader.dataset)} "
        f"validation={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )

    model = ViGResNetFusion(
        num_classes=200 if args.dataset == 'tiny-imagenet-200' else 100,
        freeze_resnet=args.freeze_resnet,
        freeze_vig=args.freeze_vig,
        proj_dim=512,
        graph_dim=128,
        graph_pool=7,
        mode=args.mode,
        use_cross_attn=args.use_cross_attention,
        resnet_backbone=args.resnet_backbone,
        vig_backbone=args.vig_backbone,
        use_glore=args.use_glore,
        glore_mid_ratio=args.glore_mid_ratio,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f}M (trainable: {trainable_params / 1e6:.2f}M)")

    loss_fn = FusionLoss(
        mode=args.mode,
        graph_loss_weight=args.graph_loss_weight,
        graph_alpha=args.graph_alpha,
        graph_beta=args.graph_beta,
        graph_k=args.graph_k,
        graph_eig_dim=args.graph_eig_dim,
        graph_tau=args.graph_tau,
        graph_stage_weights=graph_stage_weights,
        graph_use=args.graph_use,
        graph_spectral_method=args.graph_spectral_method,
        graph_kde_points=args.graph_kde_points,
        graph_kde_bandwidth=args.graph_kde_bandwidth,
        graph_kde_bandwidth_scale=args.graph_kde_bandwidth_scale,
        graph_kde_eig_jitter=args.graph_kde_eig_jitter,
        graph_distance_mode=args.graph_distance_mode,
        align_weight=args.align_weight,
        graph_anneal=args.graph_anneal,
        anneal_start=args.anneal_start,
        max_epochs=args.epochs,
        prob_aux_weight=args.prob_aux_weight,
    )

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)

    checkpoint_stem = f'{args.exp_name}_{args.dataset.replace("-", "_")}'
    best_checkpoint_path = os.path.abspath(f'best_{checkpoint_stem}.pth')
    last_checkpoint_path = os.path.abspath(args.resume or f'last_{checkpoint_stem}.pth')
    model_config = model.model_config(num_classes=200 if args.dataset == 'tiny-imagenet-200' else 100)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    train_metrics_history = []
    val_metrics_history = []
    best_acc = float('-inf')
    best_epoch = 0
    history = {
        'train_losses': train_losses, 'train_accuracies': train_accuracies,
        'val_losses': val_losses, 'val_accuracies': val_accuracies,
        'learning_rates': learning_rates, 'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
    }
    start_epoch = 1
    if resume_checkpoint is not None:
        if resume_checkpoint['model_config'] != model_config:
            raise ValueError('Checkpoint architecture does not match the constructed model')
        completed_epoch = int(resume_checkpoint['epoch'])
        if any(len(resume_checkpoint['history'][key]) != completed_epoch for key in history):
            raise ValueError('Checkpoint history length does not match its epoch')
        model.load_state_dict(resume_checkpoint['model'], strict=True)
        optim.load_state_dict(resume_checkpoint['optimizer'])
        scheduler.load_state_dict(resume_checkpoint['scheduler'])
        scaler.load_state_dict(resume_checkpoint['scaler'])
        for key in history:
            history[key].extend(resume_checkpoint['history'][key])
        best_acc = float(resume_checkpoint['best_acc'])
        best_epoch = int(resume_checkpoint['best_epoch'])
        best_checkpoint_path = os.path.join(
            os.path.dirname(last_checkpoint_path), resume_checkpoint['best_filename'])
        if not os.path.isfile(best_checkpoint_path):
            raise FileNotFoundError(f'Keep the matching best model beside the last checkpoint: {best_checkpoint_path}')
        best_saved = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        if best_saved.get('model_config') != model_config or best_saved.get('selection_split') != 'validation':
            raise ValueError('The best-model file does not match this training run')
        if float(best_saved['val_acc']) > best_acc:
            best_acc = float(best_saved['val_acc'])
            best_epoch = int(best_saved['epoch'])
        del best_saved
        rng = resume_checkpoint['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng['cuda'] is not None:
            torch.cuda.set_rng_state_all(rng['cuda'])
        start_epoch = completed_epoch + 1
        del resume_checkpoint
        if loaded_path != last_checkpoint_path and os.path.isfile(last_checkpoint_path):
            # Preserve the good backup rather than rotating a damaged primary over it.
            os.replace(last_checkpoint_path, last_checkpoint_path + '.corrupt')
        print(f'Continuing at epoch {start_epoch}; best validation={best_acc:.2f}%')
    print(f'Full checkpoint every {args.checkpoint_every} epochs: {last_checkpoint_path}')

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optim.param_groups[0]['lr']

        tr_loss, tr_acc, tr_metrics = train_one_epoch(
            model,
            train_loader,
            optim,
            loss_fn,
            device,
            epoch,
            amp=not args.no_amp,
            scaler=scaler,
        )
        va_loss, va_acc, va_metrics = evaluate(model, val_loader, loss_fn, device, epoch, amp=not args.no_amp)

        train_losses.append(tr_loss)
        train_accuracies.append(tr_acc)
        val_losses.append(va_loss)
        val_accuracies.append(va_acc)
        learning_rates.append(current_lr)
        train_metrics_history.append(tr_metrics)
        val_metrics_history.append(va_metrics)

        scheduler.step()
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = epoch
            _atomic_torch_save({
                'model': model.state_dict(),
                'acc': best_acc,
                'val_acc': best_acc,
                'epoch': best_epoch,
                'selection_split': 'validation',
                'model_config': model_config,
            }, best_checkpoint_path)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            # Drop the last batch gradients before staging the checkpoint on CPU.
            optim.zero_grad(set_to_none=True)
            _save_training_checkpoint({
                'version': 2,
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'best_filename': os.path.basename(best_checkpoint_path),
                'model_config': model_config,
                'args': vars(args).copy(),
                'history': history,
                'rng': {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }, last_checkpoint_path)
            print(f'Training checkpoint saved at epoch {epoch}')

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% { _format_epoch_metrics(tr_metrics, args.mode) }  "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.2f}% { _format_epoch_metrics(va_metrics, args.mode) }  "
            f"best={best_acc:.2f}%"
        )

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint['model'], strict=True)
    test_loss, test_acc, test_metrics = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        best_epoch,
        amp=not args.no_amp,
    )
    print(
        f"Official test evaluation (best validation epoch {best_epoch:03d}): "
        f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}% "
        f"{_format_epoch_metrics(test_metrics, args.mode)}"
    )

    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, learning_rates, args)

    training_history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'best_val_acc': best_acc,
        'best_val_epoch': best_epoch,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_metrics': test_metrics,
        'args': vars(args),
    }

    torch.save(training_history, f'training_history_{args.exp_name}_{args.dataset.replace("-", "_")}.pth')

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best validation epoch: {best_epoch}")
    print(f"Official test accuracy: {test_acc:.2f}%")
    print(f"Training curves saved to: training_curves_{args.exp_name}_{args.dataset.replace('-', '_')}.pdf")
    print(f"Training history saved to: training_history_{args.exp_name}_{args.dataset.replace('-', '_')}.pth")


if __name__ == '__main__':
    main()
