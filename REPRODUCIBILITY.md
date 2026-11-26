# Reproducibility record

## Known differences / missing evidence

1. The local trainer currently splits a stratified validation subset from the CIFAR training set (`--val-ratio 0.1` by default). The manuscript does not specify this split. This snapshot must not be presented as an exact historical training snapshot.
2. The original full training arguments, library lockfile, GPU details, and original run logs for each manuscript table row were not available in the curated inputs. The example launcher uses current defaults for graph weight (0.1), adjacency/spectrum coefficients (1.0/0.2), and no graph-weight annealing. These must be checked against original checkpoint metadata before claiming exact reproduction.
3. The published model is described as trained from scratch; the released backbone constructors request no pretrained weights. This release has not retrained the models to reproduce the manuscript metrics.
4. The manuscript reports Ours 78.75%, whereas the released patch-smoothing evaluation CSV reports 78.66% at severity zero. It reports Pure ViG-S 76.96% vs the manuscript comparison table's 76.64%, and Pure ViT 66.60% vs the table's 67.56%. The saved smoothing records are retained as-is. They must not silently substitute for the manuscript comparison table.
5. Model weights are not bundled; exact checkpoint provenance and availability are pending. Per-image predictions and paired source data permit independent verification of the saved robustness plots without loading weights.
6. The selected per-image records are predictions/probabilities with CIFAR test indices, not CIFAR image content or personal data.

## Checks performed for this release

All Python files are syntax-checked; artifact and source hashes are recorded. Plot reconstruction is checked against the released CSV. A training/model forward pass is not claimed: the host environment has a PyTorch OpenMP initialization conflict and lacks torchvision/timm. Original CIFAR training and evaluation were not rerun.

## Figure data protocol

Patch size 4; smoothing severities 0, 0.25, 0.5, 0.75, 1; seed 42; 10,000 CIFAR-100 test images. Each paired plot uses only images initially correct for both models. Table values and saved evaluation measurements are distinguished in `results/`.
