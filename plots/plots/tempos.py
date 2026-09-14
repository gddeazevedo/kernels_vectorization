import matplotlib.pyplot as plt

from plots.style import COLORS, MARKERS


def plot_tempos(df, metric, title, filename):
    _, ax = plt.subplots(figsize=(10, 6))

    for i, variante in enumerate(df["variante"].unique()):
        subset = df[df["variante"] == variante].sort_values("N")
        ax.plot(
            subset["N"],
            subset[metric],
            label=variante,
            color=COLORS.get(variante, None),
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("N (dimensão da matriz)", fontsize=12)
    ax.set_ylabel("Tempo (s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
