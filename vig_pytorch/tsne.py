import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from tqdm import tqdm
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from vig import *
from two_branch import ViGResNetFusion as _TrainingViGResNetFusion


class ViGResNetFusion(_TrainingViGResNetFusion):
    """t-SNE adapter preserving the script's historical feature return format."""

    def forward(self, x):
        logits, _, branch_features = super().forward(x)
        return logits, branch_features.get('feat_vig'), branch_features.get('feat_res')


# -------------------------
# CIFAR-100 Superclass Mapping
# -------------------------
def get_cifar100_superclass_mapping():
    """
    CIFAR-100的超类映射
    返回: fine_to_coarse (dict): 细粒度类别到超类的映射
    """
    # CIFAR-100的20个超类
    superclass_names = [
        'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
        'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
        'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
        'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
        'trees', 'vehicles_1', 'vehicles_2'
    ]

    # 每个超类包含的细粒度类别
    superclass_mapping = {
        0: [4, 30, 55, 72, 95],  # aquatic_mammals
        1: [1, 32, 67, 73, 91],  # fish
        2: [54, 62, 70, 82, 92],  # flowers
        3: [9, 10, 16, 28, 61],  # food_containers
        4: [0, 51, 53, 57, 83],  # fruit_and_vegetables
        5: [22, 39, 40, 86, 87],  # household_electrical_devices
        6: [5, 20, 25, 84, 94],  # household_furniture
        7: [6, 7, 14, 18, 24],  # insects
        8: [3, 42, 43, 88, 97],  # large_carnivores
        9: [12, 17, 37, 68, 76],  # large_man-made_outdoor_things
        10: [23, 33, 49, 60, 71],  # large_natural_outdoor_scenes
        11: [15, 19, 21, 31, 38],  # large_omnivores_and_herbivores
        12: [34, 63, 64, 66, 75],  # medium_mammals
        13: [26, 45, 77, 79, 99],  # non-insect_invertebrates
        14: [2, 11, 35, 46, 98],  # people
        15: [27, 29, 44, 78, 93],  # reptiles
        16: [36, 50, 65, 74, 80],  # small_mammals
        17: [47, 52, 56, 59, 96],  # trees
        18: [8, 13, 48, 58, 90],  # vehicles_1
        19: [41, 69, 81, 85, 89]  # vehicles_2
    }

    # 创建细粒度到超类的映射
    fine_to_coarse = {}
    for coarse_label, fine_labels in superclass_mapping.items():
        for fine_label in fine_labels:
            fine_to_coarse[fine_label] = coarse_label

    return fine_to_coarse, superclass_names


# Data Loading
# -------------------------
def build_cifar100_test(data_dir: str, batch_size: int, workers: int = 4):
    normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)
    return test_loader


# -------------------------
# Feature Extraction
# -------------------------
def extract_features(model, dataloader, device, mode='vig', max_samples=None):
    """
    提取特征和标签
    """
    model.eval()
    all_features = []
    all_labels = []
    all_predictions = []

    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            if max_samples and total_samples >= max_samples:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, feat_vig, feat_res = model(images)

            # 根据模式选择特征
            if mode == 'vig':
                if feat_vig is None:
                    raise ValueError(f"ViG features are None. Check model mode.")
                features = feat_vig
            elif mode == 'resnet':
                if feat_res is None:
                    raise ValueError(f"ResNet features are None. Check model mode.")
                features = feat_res
            elif mode in ('fusion', 'prob_fusion'):
                if feat_vig is None or feat_res is None:
                    raise ValueError(f"Fusion features are None. Check model mode.")
                features = torch.cat([feat_vig, feat_res], dim=1)
            elif mode == 'vig_raw':
                if feat_vig is None:
                    raise ValueError(f"ViG raw features are None. Check model mode.")
                features = feat_vig
            elif mode == 'resnet_raw':
                if feat_res is None:
                    raise ValueError(f"ResNet raw features are None. Check model mode.")
                features = feat_res
            else:
                raise ValueError(f"Unknown mode: {mode}")

            predictions = logits.argmax(dim=1)

            # 限制样本数量
            if max_samples:
                remaining = max_samples - total_samples
                if remaining < images.size(0):
                    features = features[:remaining]
                    labels = labels[:remaining]
                    predictions = predictions[:remaining]

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

            total_samples += features.size(0)

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)

    return features, labels, predictions


# -------------------------
# t-SNE Visualization
# -------------------------
def plot_tsne(features, labels, predictions, args):
    """
    使用t-SNE进行特征可视化，包括超类可视化
    """
    print(f"\nPerforming t-SNE dimensionality reduction...")
    print(f"Feature shape: {features.shape}")

    # 执行t-SNE
    tsne = TSNE(n_components=2, random_state=args.seed, perplexity=args.perplexity,
                n_iter=args.n_iter, verbose=1)
    features_2d = tsne.fit_transform(features)

    print(f"t-SNE completed. Embedding shape: {features_2d.shape}")

    # 计算准确率
    accuracy = (predictions == labels).mean() * 100

    # 获取超类映射
    fine_to_coarse, superclass_names = get_cifar100_superclass_mapping()

    # 将细粒度标签转换为超类标签
    superclass_labels = np.array([fine_to_coarse[label] for label in labels])
    superclass_predictions = np.array([fine_to_coarse[pred] for pred in predictions])

    # 计算超类准确率
    superclass_accuracy = (superclass_predictions == superclass_labels).mean() * 100

    # 创建PDF保存多个图
    pdf_filename = f'tsne_visualization_{args.mode}_cifar100.pdf'

    with PdfPages(pdf_filename) as pdf:
        # 图1: 按真实标签着色 (细粒度类别)
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=labels, cmap='tab20', s=10, alpha=0.6)

        plt.colorbar(scatter, ax=ax, label='Fine-grained Class Label')
        ax.set_title(f't-SNE Visualization of {args.mode.upper()} Features (True Fine-grained Labels)\n'
                     f'Accuracy: {accuracy:.2f}%', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 图2: 按预测标签着色 (细粒度类别)
        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=predictions, cmap='tab20', s=10, alpha=0.6)

        plt.colorbar(scatter, ax=ax, label='Predicted Fine-grained Class')
        ax.set_title(f't-SNE Visualization of {args.mode.upper()} Features (Predictions)\n'
                     f'Accuracy: {accuracy:.2f}%', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 图3: 标记正确和错误的预测
        fig, ax = plt.subplots(figsize=(12, 10))

        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask

        # 绘制正确预测的点
        ax.scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1],
                   c='green', s=10, alpha=0.5, label=f'Correct ({correct_mask.sum()})')

        # 绘制错误预测的点
        ax.scatter(features_2d[incorrect_mask, 0], features_2d[incorrect_mask, 1],
                   c='red', s=10, alpha=0.7, label=f'Incorrect ({incorrect_mask.sum()})')

        ax.set_title(f't-SNE Visualization - Correct vs Incorrect Predictions\n'
                     f'Mode: {args.mode.upper()}, Accuracy: {accuracy:.2f}%',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 图4: 按超类真实标签着色 (新增)
        fig, ax = plt.subplots(figsize=(14, 11))

        # 使用tab20颜色映射，因为有20个超类
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=superclass_labels, cmap='tab20', s=10, alpha=0.6)

        cbar = plt.colorbar(scatter, ax=ax, label='Superclass Label', ticks=range(20))
        cbar.ax.set_yticklabels([f'{i}' for i in range(20)], fontsize=8)

        ax.set_title(f't-SNE Visualization of {args.mode.upper()} Features (True Superclass Labels)\n'
                     f'Fine-grained Acc: {accuracy:.2f}% | Superclass Acc: {superclass_accuracy:.2f}%',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 图5: 按超类预测标签着色 (新增)
        fig, ax = plt.subplots(figsize=(14, 11))

        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                             c=superclass_predictions, cmap='tab20', s=10, alpha=0.6)

        cbar = plt.colorbar(scatter, ax=ax, label='Predicted Superclass', ticks=range(20))
        cbar.ax.set_yticklabels([f'{i}' for i in range(20)], fontsize=8)

        ax.set_title(f't-SNE Visualization of {args.mode.upper()} Features (Predicted Superclass)\n'
                     f'Fine-grained Acc: {accuracy:.2f}% | Superclass Acc: {superclass_accuracy:.2f}%',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 图6: 超类正确vs错误预测 (新增)
        fig, ax = plt.subplots(figsize=(12, 10))

        superclass_correct_mask = superclass_predictions == superclass_labels
        superclass_incorrect_mask = ~superclass_correct_mask

        # 绘制超类正确预测的点
        ax.scatter(features_2d[superclass_correct_mask, 0], features_2d[superclass_correct_mask, 1],
                   c='blue', s=10, alpha=0.5, label=f'Superclass Correct ({superclass_correct_mask.sum()})')

        # 绘制超类错误预测的点
        ax.scatter(features_2d[superclass_incorrect_mask, 0], features_2d[superclass_incorrect_mask, 1],
                   c='orange', s=10, alpha=0.7, label=f'Superclass Incorrect ({superclass_incorrect_mask.sum()})')

        ax.set_title(f't-SNE Visualization - Superclass Correct vs Incorrect Predictions\n'
                     f'Mode: {args.mode.upper()}, Superclass Accuracy: {superclass_accuracy:.2f}%',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # 图7: 详细的超类分布图，带有超类名称 (新增)
        fig, ax = plt.subplots(figsize=(16, 13))

        # 为每个超类使用不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, 20))

        for superclass_id in range(20):
            mask = superclass_labels == superclass_id
            if mask.sum() > 0:
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[superclass_id]], s=15, alpha=0.7,
                           label=f'{superclass_id}: {superclass_names[superclass_id][:20]}')

        ax.set_title(f't-SNE Visualization by Superclass ({args.mode.upper()})\n'
                     f'Superclass Accuracy: {superclass_accuracy:.2f}%',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

    print(f"\nt-SNE visualization saved to: {pdf_filename}")
    print(f"Overall Fine-grained Accuracy: {accuracy:.2f}%")
    print(f"Overall Superclass Accuracy: {superclass_accuracy:.2f}%")

    # 保存t-SNE嵌入结果
    tsne_results = {
        'features_2d': features_2d,
        'labels': labels,
        'predictions': predictions,
        'accuracy': accuracy,
        'superclass_labels': superclass_labels,
        'superclass_predictions': superclass_predictions,
        'superclass_accuracy': superclass_accuracy,
        'superclass_names': superclass_names,
    }

    results_filename = f'tsne_results_{args.mode}_cifar100.npz'
    np.savez(results_filename, **tsne_results)
    print(f"t-SNE results saved to: {results_filename}")

    # 打印每个超类的准确率统计
    print("\n" + "=" * 70)
    print("Superclass-wise Accuracy:")
    print("=" * 70)
    print(f"{'ID':<4} {'Superclass Name':<35} {'Samples':<8} {'Accuracy':<10}")
    print("-" * 70)

    for superclass_id in range(20):
        mask = superclass_labels == superclass_id
        if mask.sum() > 0:
            superclass_acc = (superclass_predictions[mask] == superclass_labels[mask]).mean() * 100
            print(f"{superclass_id:<4} {superclass_names[superclass_id]:<35} {mask.sum():<8} {superclass_acc:>6.2f}%")
    print("=" * 70)


# -------------------------
# Load Model with Flexible State Dict
# -------------------------
def _checkpoint_state(checkpoint):
    state_dict = checkpoint.get('model', checkpoint)
    if not isinstance(state_dict, dict) or not all(isinstance(k, str) for k in state_dict):
        raise ValueError("Checkpoint does not contain a valid model state_dict")
    return state_dict


def infer_model_config(checkpoint, mode=None, overrides=None):
    """Restore model construction arguments, with shape-based fallback for legacy checkpoints."""
    state_dict = _checkpoint_state(checkpoint)
    saved = checkpoint.get('model_config', {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint model_config must be a dictionary")
    config = {
        'num_classes': 100,
        'proj_dim': 512,
        'graph_dim': 128,
        'graph_pool': 7,
        'mode': mode or 'vig',
        'use_cross_attn': False,
        'resnet_backbone': '18',
        'vig_backbone': 'vig_s',
        'use_glore': False,
        'glore_mid_ratio': 0.5,
        'vit_dim': 384,
        'vit_depth': 12,
        'vit_num_heads': 6,
    }
    config.update({key: value for key, value in saved.items() if key in config})

    # Legacy checkpoints predate model_config. Infer architecture from stable tensor shapes.
    if not saved:
        if mode is None:
            has_vig = any(key.startswith('vig.') for key in state_dict)
            has_resnet = any(key.startswith('resnet.') for key in state_dict)
            has_classifier = any(key.startswith('classifier.') for key in state_dict)
            has_stages = any(key.startswith('vig_stage_projectors.') for key in state_dict)
            has_probability_heads = (
                any(key.startswith('vig_classifier.') for key in state_dict)
                and any(key.startswith('res_classifier.') for key in state_dict)
            )
            if has_vig and has_resnet and has_stages and has_probability_heads:
                config['mode'] = 'prob_fusion'
            elif has_vig and has_resnet and has_stages:
                config['mode'] = 'fusion'
            elif has_vig and has_classifier:
                config['mode'] = 'vig'
            elif has_resnet and has_classifier:
                config['mode'] = 'resnet'
            elif has_vig:
                config['mode'] = 'vig_raw'
            elif has_resnet:
                config['mode'] = 'resnet_raw'
            else:
                raise ValueError("Cannot infer model mode from checkpoint keys; pass --mode explicitly")
        if 'vig.cls_token' in state_dict or 'vig.patch_embed.weight' in state_dict:
            config['vig_backbone'] = 'vit'
            config['vit_dim'] = int(state_dict['vig.cls_token'].shape[-1])
            block_ids = {
                int(key.split('.')[2]) for key in state_dict
                if key.startswith('vig.blocks.') and key.split('.')[2].isdigit()
            }
            if block_ids:
                config['vit_depth'] = max(block_ids) + 1
        elif 'vig.pos_embed' in state_dict:
            channels = int(state_dict['vig.pos_embed'].shape[1])
            if channels == 640:
                config['vig_backbone'] = 'vig_b'
            elif channels == 320:
                config['vig_backbone'] = 'vig_s'
            else:
                raise ValueError(f"Cannot infer ViG backbone from pos_embed channels={channels}")

        if 'resnet.resnet.conv1.weight' in state_dict:
            config['use_glore'] = True
            config['resnet_backbone'] = '50' if state_dict['resnet.resnet.fc.weight'].shape[1] == 2048 else '18'
            glore_state = state_dict.get('resnet.glore2.conv_state.weight')
            if glore_state is not None:
                stage_dim = 512 if config['resnet_backbone'] == '50' else 128
                config['glore_mid_ratio'] = float(glore_state.shape[0]) / (2.0 * stage_dim)
        elif 'resnet.conv1.weight' in state_dict:
            fc_weight = state_dict.get('resnet.fc.weight')
            config['resnet_backbone'] = '50' if fc_weight is not None and fc_weight.shape[1] == 2048 else '18'

        config['use_cross_attn'] = any(key.startswith('fusion.') for key in state_dict)
        for key in ('proj_vig.weight', 'proj_res.weight'):
            if key in state_dict:
                config['proj_dim'] = int(state_dict[key].shape[0])
                break
        else:
            classifier_weight = state_dict.get('classifier.0.weight')
            if classifier_weight is not None:
                classifier_input = int(classifier_weight.shape[1])
                config['proj_dim'] = (
                    classifier_input // 2
                    if config['mode'] in ('fusion', 'prob_fusion') and not config['use_cross_attn']
                    else classifier_input
                )
        for key in ('vig_stage_projectors.0.0.weight', 'res_stage_projectors.0.0.weight'):
            if key in state_dict:
                config['graph_dim'] = int(state_dict[key].shape[0])
                break

    if mode is not None:
        config['mode'] = mode
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    if config['use_glore'] and config['mode'] != 'resnet_raw':
        raise ValueError("GloRe checkpoints are only supported with mode='resnet_raw'")
    if config['mode'] in ('fusion', 'prob_fusion') and config['vig_backbone'] == 'vit' and config['vit_depth'] < 3:
        raise ValueError("Fusion with the ViT backbone requires vit_depth >= 3")
    return config


def _required_state_keys(model, mode):
    prefixes = {
        'fusion': ('vig.', 'resnet.', 'proj_vig.', 'norm_vig.', 'proj_res.', 'norm_res.',
                   'vig_stage_projectors.', 'res_stage_projectors.', 'classifier.'),
        'prob_fusion': ('vig.', 'resnet.', 'proj_vig.', 'norm_vig.', 'proj_res.', 'norm_res.',
                        'vig_stage_projectors.', 'res_stage_projectors.', 'classifier.',
                        'vig_classifier.', 'res_classifier.'),
        'vig': ('vig.', 'proj_vig.', 'norm_vig.', 'classifier.'),
        'resnet': ('resnet.', 'proj_res.', 'norm_res.', 'classifier.'),
        'vig_raw': ('vig.',),
        'resnet_raw': ('resnet.',),
    }[mode]
    return {key for key in model.state_dict() if key.startswith(prefixes)}


def load_model_flexible(model, checkpoint_path, device, mode):
    """Load a checkpoint while rejecting missing or shape-incompatible active weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _checkpoint_state(checkpoint)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        print(f"Checkpoint info - Epoch: {checkpoint.get('epoch', 'unknown')}, "
              f"Accuracy: {checkpoint.get('acc', 'unknown')}")

    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    shape_mismatches = []
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.shape != model_state_dict[key].shape:
                shape_mismatches.append(
                    f"{key}: checkpoint {tuple(value.shape)} vs model {tuple(model_state_dict[key].shape)}"
                )
            else:
                filtered_state_dict[key] = value

    required_keys = _required_state_keys(model, mode)
    missing_required = sorted(required_keys - filtered_state_dict.keys())
    if shape_mismatches or missing_required:
        details = []
        if shape_mismatches:
            details.append("shape mismatches: " + '; '.join(shape_mismatches[:5]))
        if missing_required:
            details.append("missing active keys: " + ', '.join(missing_required[:10]))
        raise RuntimeError("Checkpoint does not match the requested model architecture (" +
                           "; ".join(details) + ")")

    allowed_legacy = {
        key for key in state_dict
        if key.startswith('vig.prediction.') and key not in model_state_dict
    }
    unexpected = sorted(set(state_dict) - set(model_state_dict) - allowed_legacy)
    if unexpected:
        raise RuntimeError(
            "Checkpoint contains unexpected model keys; architecture/configuration may be wrong: "
            + ', '.join(unexpected[:10])
        )

    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Successfully loaded {len(filtered_state_dict)} parameters")
    if allowed_legacy:
        print(f"Ignored {len(allowed_legacy)} legacy unused ViG classifier parameters")
    return model

    # -------------------------


# Main Function
# -------------------------
def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization for ViG Features')

    # 数据和模型参数
    parser.add_argument('--data', type=str, default='./data',
                        help='path to CIFAR-100 data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to trained model checkpoint (.pth file)')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['fusion', 'prob_fusion', 'vig', 'resnet', 'vig_raw', 'resnet_raw'],
                        help='which branch features to visualize (defaults to the checkpoint configuration)')
    parser.add_argument('--vig-backbone',
                        choices=['vig_s', 'vig_b', 'vit', 'convnext_tiny', 'vmamba_tiny',
                                 'swin_tiny', 'pvt_v2_b2', 'conformer_tiny'],
                        default=None)
    parser.add_argument('--resnet-backbone', choices=['18', '50'], default=None)
    parser.add_argument('--use-cross-attention', action='store_true', default=None)
    parser.add_argument('--use-glore', action='store_true', default=None)
    parser.add_argument('--glore-mid-ratio', type=float, default=None)
    parser.add_argument('--vit-dim', type=int, default=None)
    parser.add_argument('--vit-depth', type=int, default=None)
    parser.add_argument('--vit-num-heads', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for feature extraction')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')

    # t-SNE参数
    parser.add_argument('--perplexity', type=int, default=100,
                        help='perplexity parameter for t-SNE (default: 30)')
    parser.add_argument('--n-iter', type=int, default=1000,
                        help='number of iterations for t-SNE (default: 1000)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='maximum number of samples to visualize (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_config = infer_model_config(
        checkpoint,
        mode=args.mode,
        overrides={
            'vig_backbone': args.vig_backbone,
            'resnet_backbone': args.resnet_backbone,
            'use_cross_attn': args.use_cross_attention,
            'use_glore': args.use_glore,
            'glore_mid_ratio': args.glore_mid_ratio,
            'vit_dim': args.vit_dim,
            'vit_depth': args.vit_depth,
            'vit_num_heads': args.vit_num_heads,
        },
    )
    args.mode = model_config['mode']

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print("t-SNE Visualization for ViG Features on CIFAR-100")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Mode: {args.mode}")
    print(f"  Branch Backbone: {model_config['vig_backbone']}")
    print(f"  ResNet Backbone: {model_config['resnet_backbone']}")
    if model_config['vig_backbone'] == 'vit':
        print(f"  ViT Config: dim={model_config['vit_dim']} depth={model_config['vit_depth']} "
              f"heads={model_config['vit_num_heads']}")
    print(f"  GloRe: {model_config['use_glore']}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Perplexity: {args.perplexity}")
    print(f"  Iterations: {args.n_iter}")
    print(f"  Max Samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"  Random Seed: {args.seed}")

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # 加载数据
    print("\n" + "=" * 70)
    print("Loading Data...")
    print("=" * 70)
    test_loader = build_cifar100_test(args.data, args.batch_size, args.workers)
    print(f"Test set size: {len(test_loader.dataset)}")

    # 加载模型
    print("\n" + "=" * 70)
    print("Loading Model...")
    print("=" * 70)

    model = ViGResNetFusion(**model_config)

    # 使用灵活的加载方法
    model = load_model_flexible(model, args.checkpoint, device, args.mode)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # 提取特征
    print("\n" + "=" * 70)
    print("Extracting Features...")
    print("=" * 70)

    features, labels, predictions = extract_features(
        model, test_loader, device, mode=args.mode, max_samples=args.max_samples
    )

    print(f"\nExtracted features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # 计算整体准确率
    overall_accuracy = (predictions == labels).mean() * 100
    print(f"\nOverall Fine-grained Accuracy: {overall_accuracy:.2f}%")

    # 计算超类准确率
    fine_to_coarse, _ = get_cifar100_superclass_mapping()
    superclass_labels = np.array([fine_to_coarse[label] for label in labels])
    superclass_predictions = np.array([fine_to_coarse[pred] for pred in predictions])
    superclass_accuracy = (superclass_predictions == superclass_labels).mean() * 100
    print(f"Overall Superclass Accuracy: {superclass_accuracy:.2f}%")

    # t-SNE可视化
    print("\n" + "=" * 70)
    print("Generating t-SNE Visualization...")
    print("=" * 70)
    plot_tsne(features, labels, predictions, args)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  • t-SNE visualization: tsne_visualization_{args.mode}_cifar100.pdf")
    print(f"  • t-SNE results: tsne_results_{args.mode}_cifar100.npz")
    print("\nVisualization includes:")
    print(f"  • Fine-grained class visualization (100 classes)")
    print(f"  • Superclass visualization (20 superclasses)")
    print(f"  • Correct vs Incorrect predictions")
    print(f"  • Detailed superclass distribution with names")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
