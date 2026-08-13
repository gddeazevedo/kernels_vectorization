import argparse
import os
import sys
import pandas as pd
from plots.speedup import plot_speedup, plot_speedup_general, plot_compiler_comparison


def operation_config(name, label):
    return {
        "runs_csv": f"{name}_runs.csv",
        "general_csv": f"{name}_general.csv",
        "plots": [
            {
                "metric": "speedup_mean",
                "title": f"{label} — Speedup Médio por Variante",
                "filename": f"{name}_speedup_mean.png",
            },
            {
                "metric": "speedup_median",
                "title": f"{label} — Mediana do Speedup por Variante",
                "filename": f"{name}_speedup_median.png",
            },
        ],
        "general_plot": {
            "metrics": ["speedup_geral_mean", "speedup_geral_median"],
            "labels": ["Speedup Médio", "Mediana do Speedup"],
            "title": f"{label} — Speedup por Variante (Geral)",
            "filename": f"{name}_speedup_general.png",
        },
        "comparison": [
            {
                "metric": "speedup_geral_mean",
                "label": "Speedup Médio",
                "title": f"{label} — Comparação de Speedup Médio (GCC vs ICX)",
                "filename": f"{name}_comparison_mean.png",
            },
            {
                "metric": "speedup_geral_median",
                "label": "Mediana do Speedup",
                "title": f"{label} — Comparação da Mediana do Speedup (GCC vs ICX)",
                "filename": f"{name}_comparison_median.png",
            },
        ],
    }


OPERATIONS = {
    "spmv": operation_config("spmv", "SpMV"),
    "ilu0":  operation_config("ilu0", "ilu0"),
}

COMPILER_LABELS = {
    "gcc": "GCC",
    "icx": "ICX",
}


def resolve_base_dir(operation):
    base_dir = os.path.join(os.path.dirname(__file__), "..", "experiments", operation)
    return os.path.normpath(base_dir)


def list_compiler_dirs(base_dir):
    return [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]


def compiler_label(compiler):
    return COMPILER_LABELS.get(compiler, compiler.upper())


def generate_per_compiler_plots(compiler_path, compiler, config):
    compiler_general_df = None
    suffix = f" ({compiler_label(compiler)})"

    runs_csv = os.path.join(compiler_path, config["runs_csv"])
    if os.path.isfile(runs_csv):
        df_runs = pd.read_csv(runs_csv)
        for plot in config["plots"]:
            out = os.path.join(compiler_path, plot["filename"])
            plot_speedup(df_runs, plot["metric"], plot["title"] + suffix, out)
            print(f"    -> {out}")
    else:
        print(f"    Aviso: {runs_csv} não encontrado")

    general_csv = os.path.join(compiler_path, config["general_csv"])
    if os.path.isfile(general_csv):
        compiler_general_df = pd.read_csv(general_csv)
        gp = config["general_plot"]
        out = os.path.join(compiler_path, gp["filename"])
        plot_speedup_general(compiler_general_df, gp["metrics"], gp["labels"], gp["title"] + suffix, out)
        print(f"    -> {out}")
    else:
        print(f"    Aviso: {general_csv} não encontrado")

    return compiler_general_df


def generate_comparison_plots(base_dir, config, compiler_general_data):
    if len(compiler_general_data) < 2 or "comparison" not in config:
        return

    print("  Gerando gráficos de comparação entre compiladores...")
    for comp in config["comparison"]:
        out = os.path.join(base_dir, comp["filename"])
        plot_compiler_comparison(
            compiler_general_data,
            comp["metric"],
            comp["label"],
            comp["title"],
            out,
        )
        print(f"    -> {out}")


def generate_plots(operation):
    config   = OPERATIONS[operation]
    base_dir = resolve_base_dir(operation)

    if not os.path.isdir(base_dir):
        print(f"Diretório não encontrado: {base_dir}")
        sys.exit(1)

    compiler_dirs = list_compiler_dirs(base_dir)
    if not compiler_dirs:
        print(f"Nenhum subdiretório encontrado em {base_dir}")
        sys.exit(1)

    compiler_general_data = {}

    for compiler in compiler_dirs:
        compiler_path = os.path.join(base_dir, compiler)

        if not os.listdir(compiler_path):
            print(f"  Pulando {compiler}/ (vazio)")
            continue

        print(f"  Gerando gráficos para {compiler}")
        df_general = generate_per_compiler_plots(compiler_path, compiler, config)
        if df_general is not None:
            compiler_general_data[compiler] = df_general

    generate_comparison_plots(base_dir, config, compiler_general_data)
    print("Concluído!")


def main():
    parser = argparse.ArgumentParser(description="Geração de gráficos de speedup")
    parser.add_argument(
        "operation",
        choices=OPERATIONS.keys(),
        help="Operação para gerar gráficos (spmv ou ilu0)",
    )
    args = parser.parse_args()

    print(f"Gerando gráficos para: {args.operation}")
    generate_plots(args.operation)


if __name__ == "__main__":
    main()
