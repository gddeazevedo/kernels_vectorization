import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Base":       "#6c757d",
    "OpenMP":     "#dc3545",
    "AVX256":     "#0d6efd",
    "AVX512":     "#fd7e14",
    "Highway256": "#17E110",
    "Highway512": "#6610f2",
}

COMPILER_COLORS = {
    "gcc": "#4285F4",
    "icx": "#EA4335",
}


def plot_speedup(df, metric, title, filename):
    _, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "X", "P", "v", "<", ">", "h"]

    for i, variante in enumerate(df["variante"].unique()):
        subset = df[df["variante"] == variante].sort_values("N")
        ax.plot(
            subset["N"],
            subset[metric],
            label=variante,
            color=COLORS.get(variante, None),
            marker=markers[i % len(markers)],
            linewidth=2,
            markersize=6,
        )

    ax.set_xscale("log")

    ax.set_xlabel("N (dimensão da matriz)", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_speedup_general(df, metrics, labels_metrics, title, filename):
    variantes = df["variante"].values
    n_metrics = len(metrics)

    _, ax = plt.subplots(figsize=(max(10, len(variantes) * 2), 6))

    bar_width = 0.8 / n_metrics
    x = np.arange(len(variantes))

    bar_colors = ["#4285F4", "#EA4335"]

    for i, (metric, label) in enumerate(zip(metrics, labels_metrics)):
        offset = (i - n_metrics / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            df[metric].values,
            bar_width,
            label=label,
            color=bar_colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(variantes, rotation=25, ha="right", fontsize=11)
    ax.set_xlabel("Variante", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_compiler_comparison(compiler_data, metric, metric_label, title, filename):
    compilers = list(compiler_data.keys())
    n_compilers = len(compilers)

    all_variantes = []
    for df in compiler_data.values():
        for v in df["variante"].values:
            if v not in all_variantes:
                all_variantes.append(v)

    _, ax = plt.subplots(figsize=(max(10, len(all_variantes) * 2), 6))

    bar_width = 0.8 / n_compilers
    x = np.arange(len(all_variantes))

    for i, compiler in enumerate(compilers):
        df = compiler_data[compiler]
        values = []
        for v in all_variantes:
            row = df[df["variante"] == v]
            values.append(row[metric].values[0] if len(row) > 0 else 0)

        offset = (i - n_compilers / 2 + 0.5) * bar_width
        color = COMPILER_COLORS.get(compiler, f"C{i}")
        ax.bar(
            x + offset,
            values,
            bar_width,
            label=compiler,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(all_variantes, rotation=25, ha="right", fontsize=11)
    ax.set_xlabel("Variante", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
