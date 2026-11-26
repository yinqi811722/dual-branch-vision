"""Explicit reference command; not a recovered historical run configuration."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dry-run', action='store_true')
p.add_argument('--graph-distance-mode', choices=['off', 'pull-close', 'push-apart'], default='push-apart')
p.add_argument('--align-weight', type=float, default=0.2)
args, extra = p.parse_known_args()
root = Path(__file__).resolve().parents[1]
cmd = [sys.executable, str(root / 'vig_pytorch/two_branch.py'), '--data', str(root / 'data'), '--dataset', 'cifar100', '--mode', 'fusion', '--vig-backbone', 'vig_s', '--resnet-backbone', '18', '--epochs', '300', '--batch-size', '128', '--lr', '1e-4', '--weight-decay', '1e-4', '--stage-weights', 'three', '--align-weight', str(args.align_weight), '--graph-distance-mode', args.graph_distance_mode, '--graph-k', '8', '--graph-eig-dim', '8', '--graph-tau', '0.5', '--graph-spectral-method', 'eigenvalue', '--aug-color-jitter', '0.4', '--aug-random-rotation', '15', '--aug-random-grayscale', '0.2', '--aug-random-erasing', '0.25', '--aug-rand-augment-n', '2', '--aug-rand-augment-m', '10', '--seed', '42'] + extra
print(shlex.join(cmd))
if not args.dry_run:
    subprocess.run(cmd, check=True)
