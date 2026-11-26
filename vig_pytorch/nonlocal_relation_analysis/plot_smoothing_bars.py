"""Render the six matched-sample comparisons as grouped bars from source data."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    output = root / "smoothing_bar_chart"
    output.mkdir(exist_ok=True)
    data = pd.read_csv(root / "patch_mean_smoothing_final_models_full" /
                       "patch_mean_smoothing_common_clean_correct.csv")
    names = ["ResNet", "Pure ViG-S", "ResNet + ViT", "Pure ViT", "Swin-T", "Conformer-Ti"]
    colors = ["#6E91B2", "#80A5C3", "#91B2CD", "#779DBD", "#ABC7E0", "#6088AA"]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.985, bottom=.105, top=.915,
                        wspace=.105, hspace=.37)
    for i, (ax, name, color) in enumerate(zip(axes.flat, names, colors)):
        subset = data[data.comparator == name].sort_values("severity")
        assert len(subset) == 4
        x = np.arange(4)
        for offset, prefix, label, fill in [(-.19, "ours", "Ours", "#BD6265"),
                                            (.19, "comparator", name, color)]:
            values = subset[f"{prefix}_survival_accuracy"].to_numpy()
            low = subset[f"{prefix}_ci_low"].to_numpy()
            high = subset[f"{prefix}_ci_high"].to_numpy()
            ax.bar(x + offset, values, width=.36, color=fill, label=label,
                   edgecolor="#777777", linewidth=.65,
                   yerr=np.vstack([values-low, high-values]),
                   error_kw={"elinewidth": .85, "capsize": 2, "capthick": .85,
                              "ecolor": "#505050"}, zorder=3)
        n = int(subset.n_common_clean_correct.iloc[0])
        ax.set_title(f"{'ABCDEF'[i]}  Ours vs {name}\nMatched clean-correct n={n:,}",
                     loc="left", fontsize=13, pad=10)
        ax.set_xticks(x, ["25", "50", "75", "100"])
        ax.tick_params(labelbottom=True)
        ax.set_ylim(0, 120)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#e5e5e5", linewidth=.7, zorder=0)
        ax.legend(loc="upper right", frameon=False, fontsize=12, handlelength=1.2,
                  borderaxespad=.2)
    fig.supxlabel("Patch-mean smoothing strength (%)", fontsize=18, y=.025)
    fig.supylabel("Accuracy retained on matched samples (%)", fontsize=18, x=.015)
    sys.path.insert(0, str(root.parent))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, json_out=str(output / "alignment.json"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(output / "smoothing_bars.png", dpi=300, facecolor="white")
    fig.savefig(output / "smoothing_bars.pdf", facecolor="white")
    fig.savefig(output / "smoothing_bars.svg", facecolor="white")
    data[data.comparator.isin(names)].to_csv(output / "source_data.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
