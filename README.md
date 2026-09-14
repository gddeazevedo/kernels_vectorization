# Vectorization Benchmarks

Benchmarks de operações de algebra linear com vetorização (AVX256, AVX512) e paralelismo (OpenMP).

## Operacoes disponiveis

- **spmv** — Sparse Matrix-Vector Multiplication (SpMV) com BCSR
- **ilu** — Incomplete LU Factorization (em desenvolvimento)

## Dependencias

- GCC (g++) com suporte a OpenMP
- Intel oneAPI (icpx) — opcional
- CMake >= 3.10
- Google Highway (SIMD)
- Python 3 com pip (para geração de gráficos)

Todas as dependencias sao instaladas automaticamente via Docker.

## Setup

```bash
make up      # sobe o container
make bash    # acessa o shell do container
```

Para derrubar o container:

```bash
make down
```

## Compilar e rodar benchmarks

Dentro do container:

```bash
make <compilador> <operacao>
```

### Exemplos

```bash
make gcc spmv    # compila com g++ e roda benchmark SpMV
make icx spmv    # compila com icpx e roda benchmark SpMV
```

## Gerar gráficos

Após rodar os benchmarks, gere os gráficos com o CLI:

```bash
pip install pandas matplotlib --break-system-packages
python3 main.py <operacao> [--tempos]
```

Sem `--tempos`, todos os gráficos são gerados. Com `--tempos`, apenas os gráficos de tempo.

### Exemplos

```bash
python3 main.py spmv             # gera todos os gráficos do SpMV
python3 main.py ilu0             # gera todos os gráficos do ILU0
python3 main.py spmv --tempos    # gera somente os gráficos de tempo do SpMV
```

Os gráficos são salvos em `experiments/<operacao>/<compilador>/` junto aos CSVs correspondentes,
exceto os de comparação entre compiladores, que ficam em `experiments/<operacao>/`.

### Gráficos gerados por operação

- `<op>_speedup_mean.png` — speedup médio por variante em função de N
- `<op>_speedup_median.png` — mediana do speedup por variante em função de N
- `<op>_speedup_general.png` — speedup geral (média e mediana) por variante (barras)
- `<op>_tempo_mean.png` — tempo médio (s) por variante em função de N
- `<op>_tempo_median.png` — mediana do tempo (s) por variante em função de N
- `<op>_erro_max.png` — erro máximo por variante em função de N
- `<op>_comparison_mean.png` — comparação do speedup médio geral entre GCC e ICX (barras)
- `<op>_comparison_median.png` — comparação da mediana do speedup geral entre GCC e ICX (barras)

## Estrutura

```
src/            # ponto de entrada (main.cpp)
libs/
  cli/          # CLI de selecao de operacao
  bcsr/         # formato Blocked CSR
  bc_matvec/    # variantes de matvec (base, AVX256, AVX512, OpenMP, Highway)
  benchmarks/
    spmv/       # benchmarks de SpMV
    ilu/        # benchmarks de ILU (TODO)
  ilu/          # fatoracao ILU (TODO)
experiments/    # resultados dos benchmarks (CSV + gráficos)
plots/          # scripts de visualização (funções de plot)
  plots/
  main.py         # CLI para geração de gráficos
demos/          # exemplos avulsos de vetorizacao
```
