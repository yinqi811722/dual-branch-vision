# Dual-branch CNN–ViG image classification

Research materials accompanying **Dual-Branch Learning of Physical Perception and Abstract Relational Representations for Image Classification**.

Authors: Xiangrong Wang, Qi Yin, Zhenrong Ruan, Zongze Wu.
Corresponding author: Zongze Wu. Editorial submission contact: Xiangrong Wang.

## Release status

This is a curated snapshot of the available local implementation and analysis data, not a certified reconstruction of the original training runs. No new accuracy results were generated for this release. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for known gaps and differences from the manuscript. Trained weights and original per-run training logs are not included.

## Contents

- `vig_pytorch/two_branch.py`: CNN–ViG model, graph losses, training and baseline modes.
- `vig_pytorch/gcn_lib/`: graph operators; `vig.py`: ViG backbone.
- `third_party/Conformer/`: optional comparison backbone with its upstream license.
- `vig_pytorch/nonlocal_relation_analysis/`: perturbation evaluation and Figure 3 rendering.
- `results/`: values transcribed from manuscript tables (explicitly labeled as reported, not rerun results).
- `patch_mean_smoothing_final_models_full/` under the analysis directory: saved summary and paired results, plus compressed per-image prediction records.

## Environment

Use a clean Python environment and install a mutually compatible PyTorch/torchvision build for your operating system and accelerator. Then run `python -m pip install -r requirements.txt`. These are dependency declarations, not the original training environment lockfile. The plotting-only workflow needs `requirements-figures.txt` and no GPU.

## Reproduce Figure 3 from saved data

From the repository root:

```bash
python -m pip install -r requirements-figures.txt
python vig_pytorch/nonlocal_relation_analysis/plot_smoothing_bars.py
```

The script reads the released paired summary CSV and writes PDF, PNG and SVG under `vig_pytorch/nonlocal_relation_analysis/smoothing_bar_chart/`. Times New Roman is requested; systems without that font will substitute another font. The numerical input is unchanged. The private Codex-skill path was replaced with a bundled alignment checker.

## Train / evaluate

```bash
python vig_pytorch/two_branch.py --help
python scripts/train_reference.py --dry-run
python scripts/train_reference.py
```

The reference launcher explicitly applies the manuscript's described augmentation and loss settings where known, and current implementation defaults where historical settings are unavailable. It is **not** an assertion that this exact command produced the reported checkpoint. Each run writes checkpoints relative to its working directory; use separate directories for separate ablations.

For perturbation evaluation, first supply your own compatible trained checkpoints and CIFAR cache:

```bash
python vig_pytorch/nonlocal_relation_analysis/evaluate_patch_permutation.py --help
python vig_pytorch/nonlocal_relation_analysis/evaluate_patch_permutation.py --data ./data --checkpoint-dir ./checkpoints --perturbation patch-mean-smoothing --samples-per-class 100 --patch-size 4 --bootstrap 10000 --seed 42 --model-set all --exclude-model PVTv2-B2 --extra-model "Pure ViG-S=best_vig_cifar100.pth" --output-dir ./evaluation_results
```

The CIFAR dataset is available from https://www.cs.toronto.edu/~kriz/cifar.html and is not redistributed here. Do not execute untrusted pickle-based checkpoints.

## Attribution

ViG/graph source carries upstream Huawei/DeepGCNs notices; Conformer includes its upstream Apache license. See the bundled notices. No new blanket license has been applied to author-owned additions. Their reuse terms must be determined by the authors.
