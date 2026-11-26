"""Adapters that expose pyramid image backbones through the ViG interface."""

from importlib import import_module, util
from pathlib import Path
from typing import List, Optional, Tuple

import timm
import torch
from torch import nn


RAW_TIMM_BACKBONES = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "pvt_v2_b2": "pvt_v2_b2",
}
RAW_ONLY_BACKBONES = frozenset((*RAW_TIMM_BACKBONES, "conformer_tiny"))


class PyramidBackboneAdapter(nn.Module):
    """Give a multi-stage feature extractor ViG-compatible methods."""

    def __init__(self, feature_extractor: nn.Module, stage_dims: List[int], num_classes: Optional[int]):
        super().__init__()
        if len(stage_dims) != 3:
            raise ValueError(f"Expected three stage dimensions, got {stage_dims}")
        self.feature_extractor = feature_extractor
        self.stage_dims = list(stage_dims)
        self.num_features = self.stage_dims[-1]
        self.head = nn.Linear(self.num_features, num_classes) if num_classes is not None else nn.Identity()

    def forward_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        stages = list(self.feature_extractor(x))
        if len(stages) != 3:
            raise RuntimeError(f"Backbone returned {len(stages)} stages; expected 3")
        return stages

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        final_map = self.forward_stage_features(x)[-1]
        return nn.functional.adaptive_avg_pool2d(final_map, 1).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_convnext_tiny(num_classes: Optional[int] = None) -> Tuple[nn.Module, int, List[int]]:
    """Build a randomly initialized ConvNeXt-T with its final three feature stages."""
    extractor = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        features_only=True,
        out_indices=(1, 2, 3),
    )
    stage_dims = list(extractor.feature_info.channels())
    model = PyramidBackboneAdapter(extractor, stage_dims, num_classes)
    return model, stage_dims[-1], stage_dims


def _import_vmamba_backbone():
    errors = []
    for module_name in ("classification.models.vmamba", "vmamba"):
        try:
            return import_module(module_name).Backbone_VSSM
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "VMamba is not importable. Clone the official MzeroMiko/VMamba repository, "
        "build kernels/selective_scan, and add the VMamba repository root to PYTHONPATH. "
        "Import attempts: " + " | ".join(errors)
    )


def build_vmamba_tiny(num_classes: Optional[int] = None) -> Tuple[nn.Module, int, List[int]]:
    """Build the official VMamba-v2 Tiny configuration with three output stages."""
    backbone_cls = _import_vmamba_backbone()
    stage_dims = [192, 384, 768]
    extractor = backbone_cls(
        out_indices=(1, 2, 3),
        pretrained=None,
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 5, 2],
        dims=96,
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        drop_path_rate=0.2,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
    )
    model = PyramidBackboneAdapter(extractor, stage_dims, num_classes)
    return model, stage_dims[-1], stage_dims


def build_pyramid_backbone(
    backbone: str,
    num_classes: Optional[int] = None,
) -> Tuple[nn.Module, int, List[int]]:
    builders = {
        "convnext_tiny": build_convnext_tiny,
        "vmamba_tiny": build_vmamba_tiny,
    }
    try:
        builder = builders[backbone]
    except KeyError as exc:
        raise ValueError(f"Unsupported pyramid backbone: {backbone}") from exc
    return builder(num_classes=num_classes)


def build_raw_timm_classifier(backbone: str, num_classes: int) -> nn.Module:
    """Build a standard timm classifier used only by the raw baseline mode."""
    try:
        model_name = RAW_TIMM_BACKBONES[backbone]
    except KeyError as exc:
        raise ValueError(f"Unsupported raw timm backbone: {backbone}") from exc
    return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def _import_conformer_class():
    source_path = Path(__file__).resolve().parents[1] / "third_party" / "Conformer" / "conformer.py"
    if not source_path.is_file():
        raise ImportError(
            "Conformer source is missing. Expected the official implementation at "
            f"{source_path}"
        )

    spec = util.spec_from_file_location("official_conformer", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the official Conformer module from {source_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Conformer


class ConformerTinyAdapter(nn.Module):
    """Expose official Conformer-Ti features while preserving its two classifier heads."""

    conv_feature_dim = 256
    trans_feature_dim = 384

    def __init__(self, num_classes: int):
        super().__init__()
        conformer_cls = _import_conformer_class()
        self.model = conformer_cls(
            patch_size=16,
            channel_ratio=1,
            embed_dim=self.trans_feature_dim,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=num_classes,
        )
        self.num_features = self.conv_feature_dim + self.trans_feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        batch_size = x.shape[0]
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)

        x_base = model.maxpool(model.act1(model.bn1(model.conv1(x))))
        x_conv = model.conv_1(x_base, return_x_2=False)
        x_trans = model.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_trans = torch.cat([cls_tokens, x_trans], dim=1)
        x_trans = model.trans_1(x_trans)

        for index in range(2, model.fin_stage):
            block = getattr(model, f"conv_trans_{index}")
            x_conv, x_trans = block(x_conv, x_trans)

        conv_features = model.pooling(x_conv).flatten(1)
        trans_features = model.trans_norm(x_trans)[:, 0]
        return torch.cat([conv_features, trans_features], dim=1)

    def forward_head_logits(self, features: torch.Tensor) -> List[torch.Tensor]:
        conv_features = features[:, :self.conv_feature_dim]
        trans_features = features[:, self.conv_feature_dim:]
        return [
            self.model.conv_cls_head(conv_features),
            self.model.trans_cls_head(trans_features),
        ]

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return features
        head_logits = self.forward_head_logits(features)
        return head_logits[0] + head_logits[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def build_raw_classifier(backbone: str, num_classes: int) -> nn.Module:
    """Build a randomly initialized classifier used only by raw comparison modes."""
    if backbone == "conformer_tiny":
        return ConformerTinyAdapter(num_classes)
    return build_raw_timm_classifier(backbone, num_classes)
