import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plots.style import COLORS, LINESTYLES, MARKERS


def format_thousands(valor, _pos):
    return f"{int(valor):,}".replace(",", ".")


def format_scientific(valor, _pos):
    if valor == 0:
        return "0"
    return f"{valor:.1e}"


def plot_runs_metric(df, metric, ylabel, title, filename, vary_linestyle=False, sci_yticks=False):
    _, ax = plt.subplots(figsize=(10, 6))

    for i, variante in enumerate(df["variante"].unique()):
        subset = df[df["variante"] == variante].sort_values("N")
        ax.plot(
            subset["N"],
            subset[metric],
            label=variante,
            color=COLORS.get(variante, None),
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)] if vary_linestyle else "-",
            linewidth=2,
            markersize=6,
        )

    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    if sci_yticks:
        ax.yaxis.set_major_formatter(FuncFormatter(format_scientific))

    ax.set_xlabel("N (dimensão da matriz)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
