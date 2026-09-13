"""Publication-style static figures from measured experiment files."""

import json
import shutil
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, REPORTS


def run():
    comparison = json.loads((REPORTS / "comparison.json").read_text())
    benchmark = json.loads((REPORTS / "benchmark.json").read_text())
    metrics = json.loads((REPORTS / "test_metrics.json").read_text())
    training = REPORTS / "training"
    training.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in comparison:
        source = ROOT / r["run_directory"] / "results.csv"
        shutil.copy2(source, training / f"{r['model']}_v2.csv")
        config = ROOT / r["run_directory"] / "args.yaml"
        shutil.copy2(config, training / f"{r['model']}_v2_args.yaml")
        frame = pd.read_csv(source)
        axes[0].plot(frame["epoch"], frame["metrics/mAP50-95(B)"], label=r["model"])
        axes[1].plot(frame["epoch"], frame["train/box_loss"], label=r["model"])
    axes[0].set(title="Validation localization quality", xlabel="Epoch", ylabel="mAP50–95")
    axes[1].set(title="Training localization loss", xlabel="Epoch", ylabel="Box loss")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORTS / "training_comparison.png", dpi=160)
    plt.close(fig)
    times = {r["backend"]: r for r in benchmark["results"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"pytorch_fp32": "#4961a4", "onnx_fp32": "#008c72", "onnx_int8": "#c47a25"}
    for r in metrics:
        label = r["backend"]
        ap = r["test"]["metrics/mAP50-95(B)"]
        axes[0].scatter(times[label]["median_ms"], ap, s=90, color=colors[label], label=label)
        axes[1].bar(label, r["size_mb"], color=colors[label])
    axes[0].set(
        title="Measured quality vs CPU forward latency",
        xlabel="Median forward time (ms)",
        ylabel="Reference-test mAP50–95",
        ylim=(0, 1),
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].set(title="Serialized artifact size", ylabel="MB")
    axes[1].tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "deployment_tradeoff.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
