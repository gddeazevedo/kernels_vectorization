#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>

typedef struct {
    int nb;      // número de "block rows" (nós)
    int bs;      // block size vamos usar 3 
    int nnzb;    // número de blocos não nulos
    int *ia;     // tamanho nb+1, índice inicial de cada block-row em ja/vals
    int *ja;     // tamanho nnzb, coluna (block index) de cada bloco
    double *vals;// tamanho nnzb * bs * bs, blocos armazenados consecutivamente em row-major dentro do bloco
} BlockedCSR;

// Tipo para ponteiro de função matvec
typedef void (*matvec_func)(const BlockedCSR * restrict, const double * restrict, double * restrict);

struct MatvecVariant {
    const char *name;
    matvec_func func;
};


#define num_variants 6

double gs_mean[num_variants]   = {0};
double gs_median[num_variants] = {0};
int    gs_count     = 0;

static inline double wtime() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static BlockedCSR *bc_alloc(int nb, int bs, int max_nblocks) {
    BlockedCSR *A = malloc(sizeof(BlockedCSR));
    if (!A) { perror("malloc A"); exit(1); }
    A->nb = nb;
    A->bs = bs;
    A->nnzb = 0;
    A->ia = malloc((nb + 1) * sizeof(int));
    A->ja = malloc(max_nblocks * sizeof(int));
    A->vals = malloc((size_t)max_nblocks * bs * bs * sizeof(double));
    if (!A->ia || !A->ja || !A->vals) { perror("malloc arrays"); exit(1); }
    A->ia[0] = 0;
    return A;
}

static void bc_shrink_to_fit(BlockedCSR *A) {
    A->ja = realloc(A->ja, (size_t)A->nnzb * sizeof(int));
    A->vals = realloc(A->vals, (size_t)A->nnzb * A->bs * A->bs * sizeof(double));
}

static void bc_free(BlockedCSR *A) {
    if (!A) {
        return;
    }
    free(A->ia);
    free(A->ja);
    free(A->vals);
    free(A);
}

static void bc_push_block(BlockedCSR *A, int brow, int bcol, const double *block) {
    int bs = A->bs;
    int pos = A->nnzb;
    A->ja[pos] = bcol;
    memcpy(&A->vals[(size_t)pos * bs * bs], block, (size_t)bs * bs * sizeof(double));
    A->nnzb++;
    A->ia[brow + 1] = A->nnzb;
}

BlockedCSR *generate_blocked27_3x3(int nx, int ny, int nz) {
    int N = nx * ny * nz;
    int bs = 3;
    int max_blocks = N * 27;
    BlockedCSR *A = bc_alloc(N, bs, max_blocks);

    int nnz_count = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int id = i + j * nx + k * nx * ny;
                A->ia[id] = nnz_count;

                for (int dk = -1; dk <= 1; dk++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        for (int di = -1; di <= 1; di++) {
                            int ni = i + di;
                            int nj = j + dj;
                            int nk = k + dk;
                            if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) continue;
                            int nid = ni + nj * nx + nk * nx * ny;
                            double blk[9];
                            if (nid == id) {
                                for (int r = 0; r < 9; r++) blk[r] = 0.0;
                                blk[0] = 2.0; blk[4] = 2.0; blk[8] = 2.5;
                                blk[1] = blk[2] = blk[3] = blk[5] = blk[6] = blk[7] = 0.1;
                            } else {
                                for (int r = 0; r < 9; r++) blk[r] = 0.05;
                            }
                            bc_push_block(A, id, nid, blk);
                            nnz_count++;
                        }
                    }
                }
                A->ia[id + 1] = nnz_count;
            }
        }
    }
    A->nnzb = nnz_count;
    bc_shrink_to_fit(A);
    return A;
}

void bc_matvec(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs = A->bs;
    int Nblocks = A->nb;
    int total_len = Nblocks * bs;
    for (int i = 0; i < total_len; i++) y[i] = 0.0;
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];
        for (int p = row_start; p < row_end; p++) { // percorre os blocos de uma linha
            int bcol = A->ja[p];
            const double *blk = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];
            yrow[0] += blk[0] * xcol[0];
            yrow[0] += blk[1] * xcol[1];
            yrow[0] += blk[2] * xcol[2];
            
            yrow[1] += blk[3] * xcol[0];
            yrow[1] += blk[4] * xcol[1];
            yrow[1] += blk[5] * xcol[2];

            yrow[2] += blk[6] * xcol[0];
            yrow[2] += blk[7] * xcol[1];
            yrow[2] += blk[8] * xcol[2];
        }
    }
}

void bc_matvec_omp_v1(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs        = A->bs;
    int total_len = A->nb * bs;

    for (int i = 0; i < total_len; i++) {
        y[i] = 0.0;
    }

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];

        double s0 = 0.0, s1 = 0.0, s2 = 0.0;

        #pragma omp simd reduction(+:s0,s1,s2)
        for (int p = row_start; p < row_end; p++) {
            int bcol           = A->ja[p];
            const double *blk  = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];

            s0 += blk[0]*xcol[0] + blk[1]*xcol[1] + blk[2]*xcol[2];
            s1 += blk[3]*xcol[0] + blk[4]*xcol[1] + blk[5]*xcol[2];
            s2 += blk[6]*xcol[0] + blk[7]*xcol[1] + blk[8]*xcol[2];
        }

        yrow[0] += s0;
        yrow[1] += s1;
        yrow[2] += s2;
    }
}

void bc_matvec_omp_v2(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs        = A->bs;
    int total_len = A->nb * bs;

    for (int i = 0; i < total_len; i++) {
        y[i] = 0.0;
    }

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];

        for (int p = row_start; p < row_end; p++) {
            int bcol           = A->ja[p];
            const double *blk  = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];


            for (int i = 0; i < bs; i++) {
                double acc = 0.0;

                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < bs; j++) {
                    acc += blk[i*bs + j] * xcol[j];
                }

                yrow[i] += acc;
            }
        }
    }
}

void bc_matvec_omp_v3(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs        = A->bs;
    int total_len = A->nb * bs;

    for (int i = 0; i < total_len; i++) {
        y[i] = 0.0;
    }

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];

        for (int p = row_start; p < row_end; p++) {
            int bcol           = A->ja[p];
            const double *blk  = &A->vals[p * bs * bs];
            const double *xcol = &x[bcol * bs];

            #pragma omp simd 
            for (int i = 0; i < bs; i++) {
                double acc = 0.0;
                yrow[i] += blk[i*bs + 0] * xcol[0] + blk[i*bs + 1] * xcol[1] + blk[i*bs + 2] * xcol[2];
            }
        }
    }
}

#define MASK_ZERO_SLOT_3 0x8 // 0b1000 (big endian) | 0b0001 (little endian)
static inline double hadd_256(__m256d v) {
    __m256d zero = _mm256_setzero_pd();

    // v_mod = [v0, v1, v2, 0.0]
    __m256d v_mod = _mm256_blend_pd(v, zero, MASK_ZERO_SLOT_3); 
    __m256d sum_half = _mm256_add_pd(
        v_mod, 
        _mm256_permute2f128_pd(v_mod, v_mod, 0x01) 
    );

    __m128d sum_128 = _mm256_castpd256_pd128(sum_half);
    __m128d final_sum = _mm_hadd_pd(sum_128, sum_128); 
    return _mm_cvtsd_f64(final_sum);
}

void bc_matvec_avx256(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs = A->bs;
   
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];
        int p = row_start;
        int bcol = A->ja[p];
        const double *blk  = &A->vals[(size_t)p * bs * bs];
        const double *xcol = &x[(size_t)bcol * bs];

        __m256d vx = _mm256_loadu_pd(xcol);

        __m256d vb0 = _mm256_loadu_pd(blk);
        __m256d y0 = _mm256_mul_pd(vb0, vx);

        __m256d vb1 = _mm256_loadu_pd(&blk[3]);
        __m256d y1 = _mm256_mul_pd(vb1, vx);

        __m256d vb2  = _mm256_loadu_pd(&blk[6]);
        __m256d y2 = _mm256_mul_pd(vb2, vx);

        for (int p = row_start + 1; p < row_end; p++) {
            bcol = A->ja[p];
            blk  = &A->vals[(size_t)p * bs * bs];
            xcol = &x[(size_t)bcol * bs];
            __m256d vx = _mm256_loadu_pd(xcol);
            
            vb0 = _mm256_loadu_pd(blk);
            y0  = _mm256_fmadd_pd(vb0, vx, y0);
            
            vb1 = _mm256_loadu_pd(&blk[3]);
            y1 = _mm256_fmadd_pd(vb1, vx, y1);
            
            vb2  = _mm256_loadu_pd(&blk[6]);
            y2 = _mm256_fmadd_pd(vb2, vx, y2);
        }
        yrow[0] = hadd_256(y0);
        yrow[1] = hadd_256(y1);
        yrow[2] = hadd_256(y2);   
    }
}

#define MASK_SUM_FIRST_3 0x07 // 0b00000111
#define MASK_SUM_MID_3   0x38 // 0b00111000
#define MASK_SUM_LAST_2  0xC0 // 0b11000000

void bc_matvec_avx512(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs = A->bs; // = 3 neste caso

    // Índices de permutação para replicar os elementos de x
    // [x0, x1, x2, x0, x1, x2, x0, x1]
    __m512i perm_idx = _mm512_set_epi64(1, 0, 2, 1, 0, 2, 1, 0);

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];

        __m512d acc = _mm512_setzero_pd();
        double y2_extra = 0.0;

        for (int p = row_start; p < row_end; p++) {
            int bcol = A->ja[p];
            const double *blk = &A->vals[(size_t)p * bs * bs];
            const double *xcol = &x[(size_t)bcol * bs];

            // Carrega 8 elementos do bloco
            __m512d vb = _mm512_loadu_pd(blk);

            // Carrega xcol em um vetor de 256 bits
            __m256d vx_256 = _mm256_loadu_pd(xcol);
            // Replica os elementos de xcol usando permutação
            __m512d vx_512 = _mm512_broadcast_f64x4(vx_256);
            vx_512 = _mm512_permutexvar_pd(perm_idx, vx_512);

            acc = _mm512_fmadd_pd(vb, vx_512, acc);
            y2_extra += blk[8] * xcol[2];
        }

        yrow[0] = _mm512_mask_reduce_add_pd(MASK_SUM_FIRST_3, acc);
        yrow[1] = _mm512_mask_reduce_add_pd(MASK_SUM_MID_3, acc);
        yrow[2] = _mm512_mask_reduce_add_pd(MASK_SUM_LAST_2, acc) + y2_extra;
    }
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n) {
    double *tmp = malloc(n * sizeof(double));
    memcpy(tmp, arr, n * sizeof(double));
    qsort(tmp, n, sizeof(double), cmp_double);
    double m = (n % 2 == 0) ? (tmp[n/2-1] + tmp[n/2]) / 2.0 : tmp[n/2];
    free(tmp);
    return m;
}

void evaluate_bc_matvecs(int nx, int ny, int nz, int K, FILE *csv) {
    int N = nx * ny * nz;

    printf("=============================================================\n");
    printf("Avaliação: malha %dx%dx%d (N=%d, 3N=%d), K=%d iterações\n",
           nx, ny, nz, N, 3*N, K);
    printf("=============================================================\n");

    BlockedCSR *A = generate_blocked27_3x3(nx, ny, nz);

    double *x      = malloc((size_t)3 * N * sizeof(double));
    double *y_ref  = malloc((size_t)3 * N * sizeof(double));
    double *y_test = malloc((size_t)3 * N * sizeof(double));

    for (int i = 0; i < 3 * N; i++) {
        x[i] = (double)i;
    }

    struct MatvecVariant variants[] = {
        {"Escalar",   bc_matvec},
        {"AVX256",    bc_matvec_avx256},
        {"AVX512",    bc_matvec_avx512},
        {"OpenMP_v1", bc_matvec_omp_v1},
        {"OpenMP_v2", bc_matvec_omp_v2},
        {"OpenMP_v3", bc_matvec_omp_v3}
    };

    bc_matvec(A, x, y_ref); // obter y_ref para comparação

    double *sample = malloc(K * sizeof(double));  // tempos individuais
    double  means[num_variants], medians[num_variants], errors[num_variants];

    int B = 10;

    for (int v = 0; v < num_variants; v++) {
        double sum = 0.0;

        variants[v].func(A, x, y_test); // "aquecimento" para cache
 
        for (int k = 0; k < K; k++) {
            double t0 = wtime();
            variants[v].func(A, x, y_test);
            sample[k] = wtime() - t0;
            sum += sample[k];
        }

        means[v]   = sum / K;
        medians[v] = median(sample, K);

        // std_deviation = ...;

        double max_err = 0.0;
        for (int i = 0; i < 3 * N; i++) {
            double diff = fabs(y_ref[i] - y_test[i]) / fabs(y_ref[i]);
            if (diff > max_err) max_err = diff;
        }
        errors[v] = max_err;
    }

    double mean_ref   = means[0];
    double median_ref = medians[0];

    printf("%-12s %12s %9s %12s %9s %12s\n",
           "Variante", "Media(s)", "Speedup (Mean)", "Mediana(s)", "Speedup (Median)", "Erro Max");
    printf("--------------------------------------------------------------------------\n");
    for (int v = 0; v < num_variants; v++) {
        double speedup_mean   = mean_ref   / means[v];
        double speedup_median = median_ref / medians[v];

        gs_mean[v]   += 1.0 / speedup_mean;
        gs_median[v] += 1.0 / speedup_median;

        printf("%-12s %12.6f %8.2fx %12.6f %8.2fx     %12.2e\n",
               variants[v].name,
               means[v],   speedup_mean,
               medians[v], speedup_median,
               errors[v]);

        fprintf(csv, "%d,%d,%d,%d,%s,%.6f,%.4f,%.6f,%.4f,%.2e\n",
            nx, ny, nz, N,
            variants[v].name,
            means[v],   speedup_mean,
            medians[v], speedup_median,
            errors[v]);
    }

    gs_count++;
    printf("\n");

    free(sample);
    free(x);
    free(y_ref);
    free(y_test);
    bc_free(A);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("%s <inicio> <fim> <incremento> <K>\n", argv[0]);
        return 1;
    }

    int ini = atoi(argv[1]);
    int fim = atoi(argv[2]);
    int inc = atoi(argv[3]);
    int K   = atoi(argv[4]);

    if (ini <= 0 || fim < ini || inc <= 0 || K <= 0) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    const char *names[] = {
        "Escalar",
        "AVX256",
        "AVX512",
        "OpenMP_v1",
        "OpenMP_v2",
        "OpenMP_v3"
    };

    FILE *csv = fopen("resultados.csv", "w");
    fprintf(csv, "nx,ny,nz,N,variante,media_s,speedup_mean,mediana_s,speedup_median,erro_max\n");

    for (int nx = ini; nx <= fim; nx += inc) {
        evaluate_bc_matvecs(nx, nx, nx, K, csv);
    }

    fclose(csv);

    printf("=============================================================\n");
    printf("Speedup Geral (média harmônica sobre %d malhas)\n", gs_count);
    printf("=============================================================\n");
    printf("%-12s %16s %18s\n", "Variante", "Speedup (Mean)", "Speedup (Median)");
    printf("--------------------------------------------------\n");

    FILE *speedup_geral_csv = fopen("speedup_geral.csv", "w");
    fprintf(speedup_geral_csv, "\nvariante,speedup_geral_mean,speedup_geral_median\n");

    for (int v = 0; v < num_variants; v++) {
        double speedup_mean   = gs_count / gs_mean[v];
        double speedup_median = gs_count / gs_median[v];

        printf("%-12s %15.2fx %17.2fx\n",
               names[v],
               speedup_mean,
               speedup_median);

        fprintf(speedup_geral_csv, "%s,%.4f,%.4f\n",
            names[v],
            speedup_mean,
            speedup_median);
    }
    printf("\n");

    fclose(speedup_geral_csv);

    return 0;
}