COLORS = {
    "Base":    "#6c757d",
    "OpenMP":  "#dc3545",
    "AVX256":  "#6610f2",
    "AVX512":  "#fd7e14",
    "Highway": "#17E110"
}

COMPILER_COLORS = {
    "gcc": "#4285F4",
    "icx": "#EA4335",
}

MARKERS = ["o", "s", "^", "D", "X", "P", "v", "<", ">", "h"]

# Variantes com valores idênticos (erro_max = 0, por exemplo) se sobrepõem no
# gráfico; o estilo de linha mantém todas visíveis.
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
