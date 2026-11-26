"""Controlled CIFAR-100 patch-permutation test for local/global dependence.

Patch permutation preserves every 4x4 patch exactly while progressively
destroying its global spatial arrangement. The diagnostic uses a fixed,
class-balanced subset and deterministic nested perturbations per image.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


SCRIPT_DIR = Path(__file__).resolve().parent
VIG_DIR = SCRIPT_DIR.parent
PROJECT_DIR = VIG_DIR.parent
if str(VIG_DIR) not in sys.path:
    sys.path.insert(0, str(VIG_DIR))

from tsne import infer_model_config  # noqa: E402
from two_branch import ViGResNetFusion  # noqa: E402


SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)
CORE_MODEL_SPECS = (
    ("ResNet", "best_resnet_cifar100.pth"),
    ("Pure ViT", "best_vig_raw_vit_cifar100.pth"),
    ("ResNet + ViT", "best_fusion_vit_cifar100.pth"),
    ("ResNet + ViG (Ours)", "best_fusion_cifar100.pth"),
)
ADDITIONAL_MODEL_SPECS = (
    ("Swin-T", "best_vig_raw_swin_tiny_cifar100.pth"),
    ("PVTv2-B2", "best_vig_raw_pvt_v2_b2_cifar100.pth"),
    ("Conformer-Ti", "best_vig_raw_conformer_tiny_cifar100.pth"),
)


def display_model_name(model_name: str) -> str:
    """Return the concise model name used in publication figures."""
    return "Ours" if model_name == "ResNet + ViG (Ours)" else model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "patch_permutation")
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument(
        "--perturbation",
        choices=("patch-permutation", "patch-mean-smoothing"),
        default="patch-permutation",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reuse-per-image",
        type=Path,
        default=None,
        help="Reuse an existing per-image result CSV and skip model inference.",
    )
    parser.add_argument(
        "--append-per-image",
        type=Path,
        default=None,
        help="Reuse completed models from this CSV and evaluate only missing requested models.",
    )
    parser.add_argument(
        "--model-set",
        choices=("core", "all"),
        default="core",
        help="all adds Swin-T, PVTv2-B2, and Conformer-Ti; ConvNeXt-T is intentionally excluded.",
    )
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Append a compatible checkpoint, for example Graph-off=C:/weights/graph_off.pth.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        metavar="NAME",
        help="Exclude a named model from reused data, statistics, and figures.",
    )
    return parser.parse_args()


def requested_model_specs(args: argparse.Namespace):
    specs = list(CORE_MODEL_SPECS)
    if args.model_set == "all":
        specs.extend(ADDITIONAL_MODEL_SPECS)
    for item in args.extra_model:
        if "=" not in item:
            raise ValueError(f"--extra-model must use NAME=CHECKPOINT syntax: {item}")
        name, checkpoint = item.split("=", 1)
        if not name.strip() or not checkpoint.strip():
            raise ValueError(f"Invalid --extra-model value: {item}")
        specs.append((name.strip(), checkpoint.strip()))
    excluded = set(args.exclude_model)
    specs = [(name, checkpoint) for name, checkpoint in specs if name not in excluded]
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate model names requested: {names}")
    return specs


def balanced_indices(targets: list[int], samples_per_class: int, seed: int) -> list[int]:
    targets_array = np.asarray(targets)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in range(100):
        candidates = np.flatnonzero(targets_array == class_id)
        selected.extend(rng.choice(candidates, samples_per_class, replace=False).tolist())
    return sorted(selected)


def permute_patches(image: np.ndarray, patch_size: int, severity: float, seed: int) -> np.ndarray:
    if severity <= 0.0:
        return image.copy()
    height, width, channels = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("patch-size must divide the image dimensions")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = (
        image.reshape(grid_h, patch_size, grid_w, patch_size, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_h * grid_w, patch_size, patch_size, channels)
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patches))
    count = max(2, int(round(severity * len(patches))))
    selected = order[:count]
    output = patches.copy()
    output[selected] = patches[np.roll(selected, 1)]
    return (
        output.reshape(grid_h, grid_w, patch_size, patch_size, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height, width, channels)
    )


def smooth_with_patch_means(image: np.ndarray, patch_size: int, severity: float) -> np.ndarray:
    if severity <= 0.0:
        return image.copy()
    height, width, channels = image.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = (
        image.reshape(grid_h, patch_size, grid_w, patch_size, channels)
        .transpose(0, 2, 1, 3, 4)
    )
    means = patches.mean(axis=(2, 3), keepdims=True)
    smoothed = (1.0 - severity) * patches.astype(np.float64) + severity * means
    return (
        np.rint(smoothed).clip(0, 255).astype(np.uint8)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height, width, channels)
    )


class PerturbedSubset(Dataset):
    def __init__(self, base, indices, severity, patch_size, seed, transform, perturbation):
        self.base = base
        self.indices = indices
        self.severity = severity
        self.patch_size = patch_size
        self.seed = seed
        self.transform = transform
        self.perturbation = perturbation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image_index = self.indices[position]
        image = self.base.data[image_index]
        if self.perturbation == "patch-permutation":
            image = permute_patches(image, self.patch_size, self.severity, self.seed + image_index * 1009)
        else:
            image = smooth_with_patch_means(image, self.patch_size, self.severity)
        return self.transform(Image.fromarray(image)), int(self.base.targets[image_index]), image_index


def load_model(path: Path, device: torch.device) -> ViGResNetFusion:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = infer_model_config(checkpoint)
    source = checkpoint.get("model", checkpoint)
    legacy_vig_head = source.get("vig.prediction.4.weight")
    if (
        config["mode"] == "vig_raw"
        and config["vig_backbone"] in {"vig_s", "vig_b"}
        and legacy_vig_head is not None
        and int(legacy_vig_head.shape[0]) != int(config["num_classes"])
    ):
        config["num_classes"] = int(legacy_vig_head.shape[0])
        print(
            f"  legacy ViG head detected: loading {config['num_classes']} output classes "
            "without altering the checkpoint",
            flush=True,
        )
    model = ViGResNetFusion(**config)
    target = model.state_dict()
    filtered = {key: value for key, value in source.items() if key in target and value.shape == target[key].shape}
    missing = sorted(set(target) - set(filtered))
    unexpected = sorted(set(source) - set(filtered))
    if missing:
        raise RuntimeError(f"{path.name} missing active parameters: {missing[:10]}")
    if unexpected and not all(key.startswith("vig.prediction.") for key in unexpected):
        raise RuntimeError(f"{path.name} has unexpected parameters: {unexpected[:10]}")
    model.load_state_dict(filtered, strict=True)
    model.eval().to(device)
    return model


def bootstrap_mean_ci(values, rng, repetitions):
    values = np.asarray(values, dtype=np.float64)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 250):
        stop = min(start + 250, repetitions)
        sample = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[sample].mean(axis=1) * 100.0
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def summarize(results: pd.DataFrame, repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    resnet = results[results.model == "ResNet"].set_index(["image_index", "severity"])
    for model_name in results.model.unique():
        model = results[results.model == model_name].set_index(["image_index", "severity"])
        for severity in SEVERITIES:
            values = model.xs(severity, level="severity").correct.to_numpy(dtype=float)
            accuracy_low, accuracy_high = bootstrap_mean_ci(values, rng, repetitions)
            baseline_values = resnet.xs(severity, level="severity").correct.to_numpy(dtype=float)
            gain = values - baseline_values
            gain_low, gain_high = bootstrap_mean_ci(gain, rng, repetitions)
            rows.append(
                {
                    "model": model_name,
                    "severity": severity,
                    "n": len(values),
                    "accuracy": 100.0 * values.mean(),
                    "accuracy_ci_low": accuracy_low,
                    "accuracy_ci_high": accuracy_high,
                    "gain_over_resnet_pp": 100.0 * gain.mean(),
                    "gain_ci_low": gain_low,
                    "gain_ci_high": gain_high,
                }
            )
    summary = pd.DataFrame(rows)
    contrasts = []
    for model_name in results.model.unique():
        if model_name == "ResNet":
            continue
        model = results[results.model == model_name].pivot(index="image_index", columns="severity", values="correct")
        res = results[results.model == "ResNet"].pivot(index="image_index", columns="severity", values="correct")
        for severity in SEVERITIES[1:]:
            relative_retention = (model[severity] - model[0.0]) - (res[severity] - res[0.0])
            low, high = bootstrap_mean_ci(relative_retention.to_numpy(), rng, repetitions)
            contrasts.append(
                {
                    "model": model_name,
                    "severity": severity,
                    "relative_retention_pp": 100.0 * relative_retention.mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    pairwise = []
    ours_name = "ResNet + ViG (Ours)"
    if ours_name in set(results.model):
        ours = results[results.model == ours_name].set_index(["image_index", "severity"])
        for model_name in results.model.unique():
            if model_name == ours_name:
                continue
            comparator = results[results.model == model_name].set_index(["image_index", "severity"])
            for severity in SEVERITIES:
                ours_values = ours.xs(severity, level="severity").correct.to_numpy(dtype=float)
                comparator_values = comparator.xs(severity, level="severity").correct.to_numpy(dtype=float)
                difference = ours_values - comparator_values
                low, high = bootstrap_mean_ci(difference, rng, repetitions)
                pairwise.append(
                    {
                        "comparator": model_name,
                        "severity": severity,
                        "ours_minus_comparator_pp": 100.0 * difference.mean(),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
    return summary, pd.DataFrame(contrasts), pd.DataFrame(pairwise)


def summarize_common_clean_correct(results: pd.DataFrame, repetitions: int, seed: int):
    """Compare corruption survival after exactly matching clean-image correctness."""
    rng = np.random.default_rng(seed + 7919)
    wide = results.pivot(index="image_index", columns=["model", "severity"], values="correct")
    ours_name = "ResNet + ViG (Ours)"
    rows = []
    if ours_name not in set(results.model):
        return pd.DataFrame(rows)
    for comparator in results.model.unique():
        if comparator == ours_name:
            continue
        common = (wide[(ours_name, 0.0)] == 1) & (wide[(comparator, 0.0)] == 1)
        for severity in SEVERITIES[1:]:
            ours_values = wide.loc[common, (ours_name, severity)].to_numpy(dtype=float)
            comparator_values = wide.loc[common, (comparator, severity)].to_numpy(dtype=float)
            difference = ours_values - comparator_values
            ours_low, ours_high = bootstrap_mean_ci(ours_values, rng, repetitions)
            comparator_low, comparator_high = bootstrap_mean_ci(comparator_values, rng, repetitions)
            low, high = bootstrap_mean_ci(difference, rng, repetitions)
            rows.append(
                {
                    "comparator": comparator,
                    "severity": severity,
                    "n_common_clean_correct": int(common.sum()),
                    "ours_survival_accuracy": 100.0 * ours_values.mean(),
                    "ours_ci_low": ours_low,
                    "ours_ci_high": ours_high,
                    "comparator_survival_accuracy": 100.0 * comparator_values.mean(),
                    "comparator_ci_low": comparator_low,
                    "comparator_ci_high": comparator_high,
                    "paired_difference_pp": 100.0 * difference.mean(),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_common_clean_correct(common_correct: pd.DataFrame, output_dir: Path, perturbation: str):
    """Plot corruption survival after matching models on clean-image correctness."""
    if common_correct.empty:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.linewidth": 1.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    ours_color = "#0F4D92"
    baseline_color = "#3B3B3B"
    severity_colors = {0.5: "#2878B5", 0.75: "#D97706"}
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.0, 1.2]})

    resnet = common_correct[common_correct.comparator == "ResNet"].sort_values("severity")
    x = np.r_[0.0, resnet.severity.to_numpy() * 100.0]
    ours_y = np.r_[100.0, resnet.ours_survival_accuracy.to_numpy()]
    resnet_y = np.r_[100.0, resnet.comparator_survival_accuracy.to_numpy()]
    ours_low = np.r_[100.0, resnet.ours_ci_low.to_numpy()]
    ours_high = np.r_[100.0, resnet.ours_ci_high.to_numpy()]
    resnet_low = np.r_[100.0, resnet.comparator_ci_low.to_numpy()]
    resnet_high = np.r_[100.0, resnet.comparator_ci_high.to_numpy()]
    for values, low, high, color, marker, label, width in (
        (ours_y, ours_low, ours_high, ours_color, "D", "Ours", 2.8),
        (resnet_y, resnet_low, resnet_high, baseline_color, "o", "ResNet", 2.0),
    ):
        axes[0].errorbar(
            x, values, yerr=np.vstack([values - low, high - values]), color=color,
            marker=marker, linewidth=width, capsize=2.5, markersize=6.5, label=label,
        )
    n_resnet = int(resnet.n_common_clean_correct.iloc[0])
    axes[0].set_title(f"Matched clean-correct subset (n={n_resnet:,})")
    axes[0].set_xlabel("Patch-mean smoothing strength (%)")
    axes[0].set_ylabel("Accuracy retained on matched samples (%)")
    axes[0].set_xticks(np.arange(0, 101, 25))
    axes[0].set_ylim(-1, 104)
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axes[0].legend(loc="upper right")

    preferred_order = [
        "ResNet", "Pure ViG-S", "ResNet + ViT", "Pure ViT", "Swin-T", "Conformer-Ti", "PVTv2-B2"
    ]
    present = set(common_correct.comparator)
    comparators = [name for name in preferred_order if name in present]
    comparators.extend(name for name in common_correct.comparator.unique() if name not in comparators)
    y_positions = np.arange(len(comparators))[::-1]
    offsets = {0.5: 0.13, 0.75: -0.13}
    for severity in (0.5, 0.75):
        subset = common_correct[common_correct.severity == severity].set_index("comparator")
        values = np.array([subset.loc[name, "paired_difference_pp"] for name in comparators])
        low = np.array([subset.loc[name, "ci_low"] for name in comparators])
        high = np.array([subset.loc[name, "ci_high"] for name in comparators])
        axes[1].errorbar(
            values, y_positions + offsets[severity], xerr=np.vstack([values - low, high - values]),
            fmt="D" if severity == 0.5 else "o", color=severity_colors[severity],
            markersize=6.5, capsize=3, elinewidth=1.8,
            label=f"{int(severity * 100)}% smoothing",
        )
    labels = []
    for name in comparators:
        n_value = int(common_correct[common_correct.comparator == name].n_common_clean_correct.iloc[0])
        labels.append(f"{name}  (n={n_value:,})")
    axes[1].axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y_positions, labels=labels)
    axes[1].set_xlabel("Paired accuracy advantage of ours (pp)")
    axes[1].set_title("Advantage after matching clean correctness")
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axes[1].legend(loc="lower right")
    figure.tight_layout(pad=1.1, w_pad=2.4)
    prefix = "patch_permutation" if perturbation == "patch-permutation" else "patch_mean_smoothing"
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            output_dir / f"{prefix}_clean_correct_matched.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def plot_all_common_clean_correct_pairs(common_correct: pd.DataFrame, output_dir: Path, perturbation: str):
    """Draw one matched clean-correct survival panel for every comparator."""
    if common_correct.empty:
        return
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.2,
            "axes.linewidth": 1.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    ours_color = "#0F4D92"
    comparator_colors = {
        "ResNet": "#333333",
        "Pure ViG-S": "#6A3D9A",
        "ResNet + ViT": "#9A4D8E",
        "Pure ViT": "#B64342",
        "Swin-T": "#D88C00",
        "Conformer-Ti": "#168AAD",
        "PVTv2-B2": "#238B45",
        "Graph-off": "#777777",
    }
    preferred_order = [
        "ResNet", "Pure ViG-S", "ResNet + ViT", "Pure ViT", "Swin-T", "Conformer-Ti", "PVTv2-B2"
    ]
    present = set(common_correct.comparator)
    comparators = [name for name in preferred_order if name in present]
    comparators.extend(name for name in common_correct.comparator.unique() if name not in comparators)
    n_columns = 3
    n_rows = int(np.ceil(len(comparators) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(13.2, 4.2 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    panel_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for panel_index, (axis, comparator) in enumerate(zip(axes.flat, comparators)):
        subset = common_correct[common_correct.comparator == comparator].sort_values("severity")
        x = np.r_[0.0, subset.severity.to_numpy() * 100.0]
        ours_y = np.r_[100.0, subset.ours_survival_accuracy.to_numpy()]
        comparator_y = np.r_[100.0, subset.comparator_survival_accuracy.to_numpy()]
        ours_low = np.r_[100.0, subset.ours_ci_low.to_numpy()]
        ours_high = np.r_[100.0, subset.ours_ci_high.to_numpy()]
        comparator_low = np.r_[100.0, subset.comparator_ci_low.to_numpy()]
        comparator_high = np.r_[100.0, subset.comparator_ci_high.to_numpy()]
        for values, low, high, color, marker, label, width in (
            (ours_y, ours_low, ours_high, ours_color, "D", "Ours", 2.6),
            (
                comparator_y, comparator_low, comparator_high,
                comparator_colors.get(comparator, "#555555"), "o", comparator, 1.9,
            ),
        ):
            axis.errorbar(
                x, values, yerr=np.vstack([values - low, high - values]),
                color=color, marker=marker, linewidth=width, capsize=2.2,
                markersize=5.8, label=label,
            )
        n_common = int(subset.n_common_clean_correct.iloc[0])
        axis.set_title(f"{panel_labels[panel_index]}  Ours vs {comparator}\nMatched clean-correct n={n_common:,}", loc="left")
        axis.set_xticks(np.arange(0, 101, 25))
        axis.set_ylim(-1, 104)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        axis.legend(loc="upper right", fontsize=8.5, handlelength=2.4)
    for axis in axes.flat[len(comparators):]:
        axis.set_visible(False)
    figure.supxlabel("Patch-mean smoothing strength (%)", y=0.02)
    figure.supylabel("Accuracy retained on matched samples (%)", x=0.02)
    figure.tight_layout(rect=(0.035, 0.045, 1.0, 1.0), h_pad=2.0, w_pad=1.4)
    prefix = "patch_permutation" if perturbation == "patch-permutation" else "patch_mean_smoothing"
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            output_dir / f"{prefix}_clean_correct_all_pairs.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def plot(summary: pd.DataFrame, output_dir: Path, perturbation: str):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.linewidth": 1.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    styles = {
        "ResNet": ("#272727", "o", 2.0),
        "Pure ViT": ("#B64342", "^", 1.8),
        "ResNet + ViT": ("#9A4D8E", "s", 2.0),
        "ResNet + ViG (Ours)": ("#0F4D92", "D", 3.0),
        "Pure ViG-S": ("#6A3D9A", "*", 2.0),
        "Swin-T": ("#D88C00", "v", 1.8),
        "PVTv2-B2": ("#238B45", "P", 1.8),
        "Conformer-Ti": ("#168AAD", "X", 1.8),
        "Graph-off": ("#777777", "h", 1.8),
    }
    fallback = plt.get_cmap("tab10")
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    for model_index, model_name in enumerate(summary.model.unique()):
        rows = summary[summary.model == model_name].sort_values("severity")
        color, marker, width = styles.get(model_name, (fallback(model_index), "o", 1.8))
        accuracy_error = np.vstack(
            [rows.accuracy - rows.accuracy_ci_low, rows.accuracy_ci_high - rows.accuracy]
        )
        axes[0].errorbar(
            rows.severity * 100, rows.accuracy, yerr=accuracy_error,
            color=color, marker=marker, linewidth=width, capsize=2.5,
            label=display_model_name(model_name),
        )
        if model_name != "ResNet":
            gain_error = np.vstack(
                [rows.gain_over_resnet_pp - rows.gain_ci_low, rows.gain_ci_high - rows.gain_over_resnet_pp]
            )
            axes[1].errorbar(
                rows.severity * 100, rows.gain_over_resnet_pp, yerr=gain_error,
                color=color, marker=marker, linewidth=width, capsize=2.5,
                label=display_model_name(model_name),
            )
    axes[0].set_ylabel("Classification accuracy (%)")
    axes[1].set_ylabel("Accuracy gain over ResNet (pp)")
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    x_label = "Permuted patches (%)" if perturbation == "patch-permutation" else "Patch-mean smoothing strength (%)"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.set_xticks(np.arange(0, 101, 25))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axes[0].legend(loc="best", ncol=2, columnspacing=1.1)
    axes[1].legend(loc="best", ncol=2, columnspacing=1.1)
    figure.tight_layout(pad=1.2)
    filename = "patch_permutation_robustness" if perturbation == "patch-permutation" else "patch_mean_smoothing_robustness"
    for extension in ("pdf", "svg", "png"):
        figure.savefig(output_dir / f"{filename}.{extension}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_critical_severities(summary: pd.DataFrame, output_dir: Path, perturbation: str):
    """Show model separation at the two informative intervention strengths."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.linewidth": 1.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {
        "ResNet": "#272727",
        "Pure ViT": "#B64342",
        "ResNet + ViT": "#9A4D8E",
        "ResNet + ViG (Ours)": "#0F4D92",
        "Pure ViG-S": "#6A3D9A",
        "Swin-T": "#D88C00",
        "PVTv2-B2": "#238B45",
        "Conformer-Ti": "#168AAD",
        "Graph-off": "#777777",
    }
    preferred_order = [
        "ResNet + ViG (Ours)", "ResNet", "Pure ViG-S", "ResNet + ViT", "Swin-T",
        "Conformer-Ti", "Pure ViT", "PVTv2-B2", "Graph-off",
    ]
    present = set(summary.model)
    model_order = [name for name in preferred_order if name in present]
    model_order.extend(name for name in summary.model.unique() if name not in model_order)
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), sharey=True)
    y_positions = np.arange(len(model_order))[::-1]
    for axis, severity in zip(axes, (0.5, 0.75)):
        subset = summary[summary.severity == severity].set_index("model")
        for y_position, model_name in zip(y_positions, model_order):
            row = subset.loc[model_name]
            xerr = np.array(
                [[row.accuracy - row.accuracy_ci_low], [row.accuracy_ci_high - row.accuracy]]
            )
            axis.errorbar(
                row.accuracy, y_position, xerr=xerr,
                fmt="D" if model_name == "ResNet + ViG (Ours)" else "o",
                markersize=7.5 if model_name == "ResNet + ViG (Ours)" else 6.0,
                color=colors.get(model_name, "#555555"), capsize=3,
                elinewidth=2.4 if model_name == "ResNet + ViG (Ours)" else 1.6,
            )
        axis.set_title(f"{int(severity * 100)}% patch-mean smoothing")
        axis.set_xlabel("Classification accuracy (%)")
        axis.set_yticks(y_positions, labels=[display_model_name(name) for name in model_order])
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    axes[0].set_ylabel("Model")
    figure.tight_layout(pad=1.1, w_pad=2.0)
    prefix = "patch_permutation" if perturbation == "patch-permutation" else "patch_mean_smoothing"
    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            output_dir / f"{prefix}_critical_severities.{extension}",
            dpi=300, bbox_inches="tight", facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_specs = requested_model_specs(args)
    device = torch.device(args.device)
    base = datasets.CIFAR100(root=args.data, train=False, download=False)
    indices = balanced_indices(base.targets, args.samples_per_class, args.seed)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    if args.reuse_per_image is not None and args.append_per_image is not None:
        raise ValueError("Use only one of --reuse-per-image and --append-per-image")
    requested_names = {name for name, _ in model_specs}
    if args.reuse_per_image is not None:
        results = pd.read_csv(args.reuse_per_image)
        results = results[results.model.isin(requested_names)].copy()
    else:
        if args.append_per_image is not None:
            prior = pd.read_csv(args.append_per_image)
            prior = prior[prior.model.isin(requested_names)].copy()
            completed_models = set(prior.model.unique())
            rows = prior.to_dict("records")
        else:
            completed_models = set()
            rows = []
        for model_name, checkpoint_name in model_specs:
            if model_name in completed_models:
                print(f"Reusing {model_name} from {args.append_per_image}", flush=True)
                continue
            print(f"Loading {model_name}: {checkpoint_name}", flush=True)
            checkpoint_path = Path(checkpoint_name)
            if not checkpoint_path.is_absolute():
                checkpoint_path = args.checkpoint_dir / checkpoint_path
            model = load_model(checkpoint_path, device)
            with torch.inference_mode():
                for severity in SEVERITIES:
                    dataset = PerturbedSubset(
                        base, indices, severity, args.patch_size, args.seed, transform, args.perturbation
                    )
                    loader = DataLoader(
                        dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                    )
                    for images, labels, image_indices in loader:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        logits, _, _ = model(images)
                        predictions = logits.argmax(dim=1)
                        log_probabilities = torch.log_softmax(logits, dim=1)
                        true_log_probability = log_probabilities.gather(1, labels[:, None]).squeeze(1)
                        for position in range(len(labels)):
                            rows.append(
                                {
                                    "model": model_name,
                                    "severity": severity,
                                    "image_index": int(image_indices[position]),
                                    "true_label": int(labels[position]),
                                    "prediction": int(predictions[position]),
                                    "correct": int(predictions[position] == labels[position]),
                                    "true_log_probability": float(true_log_probability[position]),
                                }
                            )
                    print(f"  severity={severity:.2f} complete", flush=True)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results = pd.DataFrame(rows)
    summary, contrasts, pairwise = summarize(results, args.bootstrap, args.seed)
    common_correct = summarize_common_clean_correct(results, args.bootstrap, args.seed)
    prefix = "patch_permutation" if args.perturbation == "patch-permutation" else "patch_mean_smoothing"
    results.to_csv(args.output_dir / f"{prefix}_per_image.csv", index=False, float_format="%.8f")
    summary.to_csv(args.output_dir / f"{prefix}_summary.csv", index=False, float_format="%.6f")
    contrasts.to_csv(args.output_dir / f"{prefix}_retention_contrasts.csv", index=False, float_format="%.6f")
    pairwise.to_csv(args.output_dir / f"{prefix}_ours_pairwise.csv", index=False, float_format="%.6f")
    common_correct.to_csv(
        args.output_dir / f"{prefix}_common_clean_correct.csv", index=False, float_format="%.6f"
    )
    plot(summary, args.output_dir, args.perturbation)
    plot_critical_severities(summary, args.output_dir, args.perturbation)
    plot_common_clean_correct(common_correct, args.output_dir, args.perturbation)
    plot_all_common_clean_correct_pairs(common_correct, args.output_dir, args.perturbation)
    metadata = {
        "samples": len(indices),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "patch_size": args.patch_size,
        "perturbation": args.perturbation,
        "severities": SEVERITIES,
        "models": [name for name, _ in model_specs],
        "interpretation": (
            "Patch contents are preserved; their global arrangement is progressively destroyed."
            if args.perturbation == "patch-permutation"
            else "Within-patch texture is progressively removed while patch location and mean colour are preserved."
        ),
    }
    (args.output_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nAccuracy summary")
    print(summary.pivot(index="model", columns="severity", values="accuracy").to_string(float_format=lambda x: f"{x:.2f}"))
    print("\nRelative retention versus clean")
    print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nPaired accuracy difference: ours minus comparator")
    print(pairwise.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nClean-correct-matched corruption survival")
    print(common_correct.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
