# Vectorization Benchmarks

Benchmarks de operações de algebra linear com vetorização (AVX256, AVX512, Google Highway, OpenMP), sobre matrizes esparsas no formato BCSR com blocos 3x3.

## Operacoes disponiveis

- **spmv** — Sparse Matrix-Vector Multiplication (SpMV) com BCSR
- **ilu0** — Fatoracao ILU(0) por blocos

Cada operacao e medida em 5 variantes: `Base`, `OpenMP`, `AVX256`, `AVX512` e `Highway`.

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

Dentro do container, **a partir da raiz do repositorio** (os CSVs sao gravados em caminho
relativo ao diretorio atual):

```bash
make <compilador> <operacao>
```

`<compilador>` e `gcc` ou `icx`; `<operacao>` e `spmv` ou `ilu0`. O alvo recompila do zero
(`make clean` implicito) e ja executa o benchmark.

### Exemplos

```bash
make gcc spmv    # compila com g++ e roda benchmark de SpMV
make icx spmv    # compila com icpx e roda benchmark de SpMV
make gcc ilu0    # compila com g++ e roda benchmark de ILU(0)
make icx ilu0    # compila com icpx e roda benchmark de ILU(0)
```

Para chamar o binario direto, sem passar pelo Makefile:

```bash
./build/kernels_vectorization <operacao> <compilador>
```

O compilador e apenas um rotulo: define o subdiretorio em que os resultados sao gravados,
nao a forma como o binario foi compilado.

### Malhas e saidas

As malhas sao cubicas (`nx = ny = nz`), com `N = nx*ny*nz`, e cada ponto e medido em 20
rodadas:

| Operacao | nx        | N                   |
|----------|-----------|---------------------|
| spmv     | 20 a 200  | 8.000 a 8.000.000   |
| ilu0     | 20 a 160  | 8.000 a 4.096.000   |

Cada execucao grava em `experiments/<operacao>/<compilador>/`:

- `<op>_runs.csv` — uma linha por malha e variante:
  `N,nx,ny,nz,variante,media_s,speedup_mean,mediana_s,speedup_median,erro_max`
- `<op>_general.csv` — media geometrica dos speedups sobre todas as malhas:
  `variante,speedup_geral_mean,speedup_geral_median`

## Gerar gráficos

Após rodar os benchmarks, **a partir do diretorio `plots/`**:

```bash
cd plots
pip install -r requirements.txt --break-system-packages
python3 main.py <operacao> [--tempos]
```

Sem `--tempos`, todos os gráficos são gerados. Com `--tempos`, apenas os gráficos de tempo.

### Exemplos

```bash
python3 main.py spmv             # gera todos os gráficos do SpMV
python3 main.py ilu0             # gera todos os gráficos do ILU(0)
python3 main.py spmv --tempos    # gera somente os gráficos de tempo do SpMV
```

Os compiladores nao sao passados na linha de comando: o script percorre todos os
subdiretorios de `experiments/<operacao>/` e gera um conjunto de gráficos para cada um.

Os gráficos são salvos em `experiments/<operacao>/<compilador>/` junto aos CSVs correspondentes,
exceto os de comparação entre compiladores, que ficam em `experiments/<operacao>/`.

### Gráficos gerados por operação

A partir de `<op>_runs.csv`, para cada compilador:

- `<op>_speedup_mean.png` — speedup médio por variante em função de N
- `<op>_speedup_median.png` — mediana do speedup por variante em função de N
- `<op>_tempo_mean.png` — tempo médio (s) por variante em função de N
- `<op>_tempo_median.png` — mediana do tempo (s) por variante em função de N
- `<op>_erro_max.png` — erro máximo por variante em função de N

A partir de `<op>_general.csv`, para cada compilador:

- `<op>_speedup_general.png` — speedup geral (média e mediana) por variante (barras)

Comparando os dois compiladores, em `experiments/<operacao>/`:

- `<op>_comparison_mean.png` — comparação do speedup médio geral entre GCC e ICX (barras)
- `<op>_comparison_median.png` — comparação da mediana do speedup geral entre GCC e ICX (barras)

## Estrutura

```
src/            # ponto de entrada (main.cpp)
libs/
  cli/          # CLI de selecao de operacao e compilador
  bcsr/         # formato Blocked CSR (blocos 3x3)
  block_ops/    # operacoes elementares de bloco (inversao, matmat, matsub)
  spmv/         # variantes de SpMV (base, OpenMP, AVX256, AVX512, Highway)
  ilu0/         # variantes de ILU(0) (base, OpenMP, AVX256, AVX512, Highway)
  utils/        # tempo, mediana, criacao de diretorios
  benchmarks/
    benchmark_base/   # laco de medicao e escrita dos CSVs
    spmv/             # benchmark de SpMV
    ilu0/             # benchmark de ILU(0)
experiments/    # resultados dos benchmarks (CSV + gráficos)
plots/
  main.py       # CLI para geração de gráficos
  plots/
    runs.py     # gráficos de tempo e erro (a partir de *_runs.csv)
    speedup.py  # gráficos de speedup e comparacao entre compiladores
    style.py    # cores, marcadores e estilos de linha compartilhados
  requirements.txt
scratch/        # exemplos avulsos de vetorizacao (fora do build principal)
```
