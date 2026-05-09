"""
Plotting utilities to reproduce paper-style figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

import config


def plot_figure4_reproduction(results: Dict):
    """
    Reproduce CLIP paper Figure 4 style:
    Left: Delta Score horizontal bar chart (relative to ResNet).
    Right: Absolute accuracy grouped bar chart with Wilson 95% CI.
    """
    datasets = list(results.keys())

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # --- Subplot 1: Delta Score (relative to ResNet) ---
    y_pos = np.arange(len(datasets))
    bar_height = 0.3

    delta_ctx = []
    delta_prm = []
    for dataset in datasets:
        resnet_acc = results[dataset]["ResNet (Supervised)"]["top1"]
        ctx_acc = results[dataset]["CLIP + Contextless"]["top1"]
        prm_acc = results[dataset]["CLIP + Prompted"]["top1"]
        delta_ctx.append((ctx_acc - resnet_acc) * 100)
        delta_prm.append((prm_acc - resnet_acc) * 100)

    colors_ctx = ["#e74c3c" if x < 0 else "#3498db" for x in delta_ctx]
    colors_prm = ["#c0392b" if x < 0 else "#2ecc71" for x in delta_prm]

    ax1.barh(
        y_pos - bar_height / 2,
        delta_ctx,
        bar_height,
        label="CLIP + Contextless",
        color=colors_ctx,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )
    ax1.barh(
        y_pos + bar_height / 2,
        delta_prm,
        bar_height,
        label="CLIP + Prompted",
        color=colors_prm,
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
    )

    ax1.axvline(x=0, color="black", linestyle="-", linewidth=2, alpha=0.8)

    for i, (v1, v2) in enumerate(zip(delta_ctx, delta_prm)):
        ha1 = "right" if v1 < 0 else "left"
        offset1 = -0.5 if v1 < 0 else 0.5
        ax1.text(
            v1 + offset1,
            i - bar_height / 2,
            f"{v1:+.1f}",
            va="center",
            ha=ha1,
            fontsize=11,
            fontweight="bold",
            color="#e74c3c" if v1 < 0 else "#2980b9",
        )

        ha2 = "right" if v2 < 0 else "left"
        offset2 = -0.5 if v2 < 0 else 0.5
        ax1.text(
            v2 + offset2,
            i + bar_height / 2,
            f"{v2:+.1f}",
            va="center",
            ha=ha2,
            fontsize=11,
            fontweight="bold",
            color="#c0392b" if v2 < 0 else "#27ae60",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(datasets, fontsize=13, fontweight="bold")
    ax1.set_xlabel("Delta Score (%) vs ResNet Supervised", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Figure 4 Style: Prompt Engineering Impact\n(Relative Performance Gap)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax1.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)
    ax1.set_xlim(min(delta_ctx + delta_prm) - 5, max(delta_ctx + delta_prm) + 5)

    # --- Subplot 2: Absolute Accuracy with Wilson CI ---
    x_pos = np.arange(len(datasets))
    width = 0.25

    resnet_accs = [results[d]["ResNet (Supervised)"]["top1"] * 100 for d in datasets]
    ctx_accs = [results[d]["CLIP + Contextless"]["top1"] * 100 for d in datasets]
    prm_accs = [results[d]["CLIP + Prompted"]["top1"] * 100 for d in datasets]

    def make_err(dataset, key):
        return [
            [
                (results[d][key]["top1"] - results[d][key]["top1_lower"]) * 100
                for d in datasets
            ],
            [
                (results[d][key]["top1_upper"] - results[d][key]["top1"]) * 100
                for d in datasets
            ],
        ]

    resnet_err = make_err(datasets, "ResNet (Supervised)")
    ctx_err = make_err(datasets, "CLIP + Contextless")
    prm_err = make_err(datasets, "CLIP + Prompted")

    ax2.bar(
        x_pos - width,
        resnet_accs,
        width,
        label="ResNet (Supervised)",
        color="#e74c3c",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
        yerr=np.array(resnet_err),
        capsize=4,
        error_kw={"linewidth": 1.5},
    )
    ax2.bar(
        x_pos,
        ctx_accs,
        width,
        label="CLIP + Contextless",
        color="#3498db",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
        yerr=np.array(ctx_err),
        capsize=4,
        error_kw={"linewidth": 1.5},
    )
    ax2.bar(
        x_pos + width,
        prm_accs,
        width,
        label="CLIP + Prompted",
        color="#2ecc71",
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2,
        yerr=np.array(prm_err),
        capsize=4,
        error_kw={"linewidth": 1.5},
    )

    for bars in ax2.containers:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 3,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax2.set_xlabel("Dataset", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=14, fontweight="bold")
    ax2.set_title(
        "Absolute Accuracy Comparison\n(9 Bars with Wilson 95% CI)",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(datasets, fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11, loc="lower right", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{config.SAVE_DIR}/figure4_complete_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_robustness_curves(clip_res: Dict, resnet_res: Dict):
    """
    Plot 2x2 grid: accuracy vs severity for each corruption type.
    """
    corruption_types = ["gaussian", "blur", "pixelate", "contrast"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, corruption in enumerate(corruption_types):
        ax = axes[idx]
        severities = range(1, 6)

        clip_accs = [
            clip_res["corrupted"][corruption][s]["accuracy"] * 100 for s in severities
        ]
        resnet_accs = [
            resnet_res["corrupted"][corruption][s]["accuracy"] * 100
            for s in severities
        ]

        ax.plot(severities, clip_accs, "o-", label="CLIP Zero-Shot", linewidth=2)
        ax.plot(severities, resnet_accs, "s-", label="ResNet Supervised", linewidth=2)

        ax.axhline(
            y=clip_res["clean"]["accuracy"] * 100,
            color="#2ecc71",
            linestyle="--",
            alpha=0.5,
            label="CLIP Clean",
        )
        ax.axhline(
            y=resnet_res["clean"]["accuracy"] * 100,
            color="#e74c3c",
            linestyle="--",
            alpha=0.5,
            label="ResNet Clean",
        )

        ax.set_xlabel("Severity Level", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"{corruption.capitalize()} Corruption", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(severities))

    plt.suptitle(
        "Robustness Analysis: CLIP vs ResNet on CIFAR-10-C",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{config.SAVE_DIR}/robustness_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_finegrained_comparison(pets_res: Dict, flowers_res: Dict):
    """
    Side-by-side bar chart for fine-grained prompt strategies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    datasets = [("Oxford-IIIT Pets", pets_res, ax1), ("Flowers102", flowers_res, ax2)]
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    for idx, (name, results, ax) in enumerate(datasets):
        strategies = ["Single_Generic", "Generic_Ensemble", "Domain_Specific"]
        x_pos = np.arange(len(strategies))
        top1_accs = [results[s]["top1"] * 100 for s in strategies]

        errors_lower = [
            (results[s]["top1"] - results[s]["top1_ci"][0]) * 100 for s in strategies
        ]
        errors_upper = [
            (results[s]["top1_ci"][1] - results[s]["top1"]) * 100 for s in strategies
        ]

        bars = ax.bar(
            x_pos,
            top1_accs,
            yerr=[errors_lower, errors_upper],
            capsize=10,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1.2,
        )

        for bar, acc in zip(bars, top1_accs):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=13, fontweight="bold")
        ax.set_title(f"{name}\n(Fine-grained Classification)", fontsize=14, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Single\nGeneric", "Generic\nEnsemble", "Domain\nSpecific"], fontsize=11)
        ax.set_ylim(0, max(top1_accs) * 1.2)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Annotate delta between Single and Domain-Specific
        single_acc = results["Single_Generic"]["top1"] * 100
        domain_acc = results["Domain_Specific"]["top1"] * 100
        delta = domain_acc - single_acc
        ax.annotate(
            f"Delta = {delta:+.1f}%",
            xy=(2, domain_acc),
            xytext=(2.3, domain_acc + 2),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=12,
            color="red",
            fontweight="bold",
        )

    plt.suptitle(
        "Fine-grained Classification: Impact of Prompt Engineering",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f"{config.SAVE_DIR}/finegrained_prompt_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()