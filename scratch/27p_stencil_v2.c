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


static inline double wtime() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

void bc_draw( BlockedCSR *A ) {
   int nb = A->nb;
   for( int i = 0; i < nb; i++ ) {
      
      for( int j = 0; j < nb; j++ ) {
         unsigned char isBlock = 0;
         for( int k = A->ia[ i ]; k < A->ia[ i + 1 ]; k++ ) {
            if( j == A->ja[ k ] ) {
               printf("[X]");
               isBlock = 1;
               break;
            }
                        
         }  
         if( isBlock == 0 ) {
            printf("   ");
         } 
      }  
      printf("\n");
      
   }

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
    if (!A) return;
    free(A->ia);
    free(A->ja);
    free(A->vals);
    free(A);
}

static void bc_push_block(BlockedCSR *A, int brow, int bcol, const double *bloc) {
    int bs = A->bs;
    int pos = A->nnzb;
    A->ja[pos] = bcol;
    memcpy(&A->vals[(size_t)pos * bs * bs], bloc, (size_t)bs * bs * sizeof(double));
    A->nnzb++;
    A->ia[brow + 1] = A->nnzb; // atualiza final da linha; será sobrescrito à medida que adicionamos
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
                                // diagonal do nó: maiores valores
                                for (int r = 0; r < 9; r++) blk[r] = 0.0;
                                blk[0] = 2.0; blk[4] = 2.0; blk[8] = 2.5; // diag
                                // pequenos acoplamentos
                                blk[1] = blk[2] = blk[3] = blk[5] = blk[6] = blk[7] = 0.1;
                            } else {
                                // vizinho: valores menores
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
    // reduz a memoria alocada!
    bc_shrink_to_fit(A);
    return A;
}

void bc_matvec(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs = A->bs;
    int Nblocks = A->nb;
    int total_len = Nblocks * bs;
    double acc = 0.0;
    for (int i = 0; i < total_len; i++) y[i] = 0.0;
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow + 1];
        double *yrow  = &y[brow * bs];
        for (int p = row_start; p < row_end; p++) {
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



void bc_matvec_avx3_o(const BlockedCSR *A, const double *x, double *y) {
    int bs = A->bs;           // = 3 neste caso
    int Nblocks = A->nb;

    int total_len = Nblocks * bs;
    for (int i = 0; i < total_len; i++) 
        y[i] = 0.0;

    const __m256d zero = _mm256_set1_pd(0.0);

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];

        for (int p = row_start; p < row_end; p++) {

            int bcol = A->ja[p];
            const double *blk  = &A->vals[(size_t)p * bs * bs];
            const double *xcol = &x[(size_t)bcol * bs];

            // Carrega x + [0]
            __m256d vx = _mm256_set_pd(0.0, xcol[2], xcol[1], xcol[0]);

            //for (int i = 0; i < bs; i++) {

                __m256d vb0 = _mm256_set_pd(0.0,
                                           blk[2],
                                           blk[1],
                                           blk[0]);
                __m256d vb1 = _mm256_set_pd(0.0,
                                           blk[5],
                                           blk[4],
                                           blk[3]);
                __m256d vb2 = _mm256_set_pd(0.0,
                                           blk[8],
                                           blk[7],
                                           blk[6]);

                __m256d prod = _mm256_mul_pd(vb0, vx);

                // horizontal add
                __m128d low = _mm256_castpd256_pd128(prod);
                __m128d high = _mm256_extractf128_pd(prod, 1);
                __m128d sum2 = _mm_add_pd(low, high);
                __m128d sum1 = _mm_hadd_pd(sum2, sum2);

                double acc = _mm_cvtsd_f64(sum1);

                yrow[0] += acc;

                prod = _mm256_mul_pd(vb1, vx);

                low = _mm256_castpd256_pd128(prod);
                high = _mm256_extractf128_pd(prod, 1);
                sum2 = _mm_add_pd(low, high);
                sum1 = _mm_hadd_pd(sum2, sum2);

                acc = _mm_cvtsd_f64(sum1);

                yrow[1] += acc;



                prod = _mm256_mul_pd(vb2, vx);

                // horizontal add
                low = _mm256_castpd256_pd128(prod);
                high = _mm256_extractf128_pd(prod, 1);
                sum2 = _mm_add_pd(low, high);
                sum1 = _mm_hadd_pd(sum2, sum2);

                acc = _mm_cvtsd_f64(sum1);

                yrow[2] += acc;
            //}
        }
    }
}

static inline double hadd_256_b(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);        // v0, v1
    __m128d hi = _mm256_extractf128_pd(v, 1);      // v2, v3
    // (v0 + v1)
    __m128d s01 = _mm_hadd_pd(lo, lo);
    
    // soma só o v2
    __m128d s012 = _mm_add_sd(s01, hi);

    return _mm_cvtsd_f64(s012);
}

// máscara para o blend, onde é 1 pega o elemento do vetor zero, onde é 0 pega do vetor original
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

void bc_matvec_avx3_pd(const BlockedCSR *A, const double *x, double *y) {
    // Como bs é sempre 3, ignoramos a variável bs dentro do kernel.
    int Nblocks = A->nb;
    int total_len = Nblocks * 3; // bs=3

    // Inicialização de y
    for (int i = 0; i < total_len; i++) 
        y[i] = 0.0;
    
    // Início do MatVec Blocked-CSR
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        // yrow = &y[brow * 3];
        // Ponteiro para o início da linha atual de y
        // similar a um slice y[brow * 3, :]
        double *yrow = &y[(size_t)brow * 3];

        // Inicializa 3 acumuladores AVX: yrow[0], yrow[1], yrow[2]
        __m256d y_acc0 = _mm256_setzero_pd();
        __m256d y_acc1 = _mm256_setzero_pd();
        __m256d y_acc2 = _mm256_setzero_pd();

        // Loop sobre os blocos não-nulos na linha (p)
        for (int p = row_start; p < row_end; p++) {

            int bcol = A->ja[p];
            // O bloco 3x3 é de 9 doubles.
            const double *blk  = &A->vals[(size_t)p * 9]; 
            // O vetor xcol é de 3 doubles.
            const double *xcol = &x[(size_t)bcol * 3];         

            // --------------------------------------------------------
            // Loop Interno Desdobrado (Unrolled): j = 0, 1, 2
            // --------------------------------------------------------

            // --- J = 0 --- (Multiplica pela primeira componente de x: xcol[0])
            __m256d vx0 = _mm256_broadcast_sd(&xcol[0]);
            
            // A[0,0] = blk[0], A[1,0] = blk[3], A[2,0] = blk[6]
            y_acc0 = _mm256_fmadd_pd(_mm256_set1_pd(blk[0]), vx0, y_acc0); // y[0] += A[0,0] * x[0]
            y_acc1 = _mm256_fmadd_pd(_mm256_set1_pd(blk[3]), vx0, y_acc1); // y[1] += A[1,0] * x[0]
            y_acc2 = _mm256_fmadd_pd(_mm256_set1_pd(blk[6]), vx0, y_acc2); // y[2] += A[2,0] * x[0]

            // --- J = 1 --- (Multiplica pela segunda componente de x: xcol[1])
            __m256d vx1 = _mm256_broadcast_sd(&xcol[1]); // carrega direto da memoria
            // __m256d vx1 = _mm256_set1_pd(xcol[1]); // carrega em um registrador para depois fazer broadcast

            // A[0,1] = blk[1], A[1,1] = blk[4], A[2,1] = blk[7]
            y_acc0 = _mm256_fmadd_pd(_mm256_set1_pd(blk[1]), vx1, y_acc0); // y[0] += A[0,1] * x[1]
            y_acc1 = _mm256_fmadd_pd(_mm256_set1_pd(blk[4]), vx1, y_acc1); // y[1] += A[1,1] * x[1]
            y_acc2 = _mm256_fmadd_pd(_mm256_set1_pd(blk[7]), vx1, y_acc2); // y[2] += A[2,1] * x[1]
            
            // --- J = 2 --- (Multiplica pela terceira componente de x: xcol[2])
            __m256d vx2 = _mm256_broadcast_sd(&xcol[2]);

            // A[0,2] = blk[2], A[1,2] = blk[5], A[2,2] = blk[8]
            y_acc0 = _mm256_fmadd_pd(_mm256_set1_pd(blk[2]), vx2, y_acc0); // y[0] += A[0,2] * x[2]
            y_acc1 = _mm256_fmadd_pd(_mm256_set1_pd(blk[5]), vx2, y_acc1); // y[1] += A[1,2] * x[2]
            y_acc2 = _mm256_fmadd_pd(_mm256_set1_pd(blk[8]), vx2, y_acc2); // y[2] += A[2,2] * x[2]

        } // Fim do loop p (blocos na linha)

        // Soma Horizontal (HADD) final
        yrow[0] += hadd_256(y_acc0);
        yrow[1] += hadd_256(y_acc1);
        yrow[2] += hadd_256(y_acc2);
    }
}


void bc_matvec_avx3_p(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs = A->bs;           // = 3 neste caso
    int Nblocks = A->nb;

    int total_len = Nblocks * bs;
    for (int i = 0; i < total_len; i++) 
        y[i] = 0.0;

    const __m256d zero = _mm256_set1_pd(0.0);

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];

        for (int p = row_start; p < row_end; p++) {

            int bcol = A->ja[p];
            const double *blk  = &A->vals[(size_t)p * bs * bs];
            const double *xcol = &x[(size_t)bcol * bs];

            // Carrega x + [0]
            __m256d vx = _mm256_set_pd(0.0, xcol[2], xcol[1], xcol[0]);

            //for (int i = 0; i < bs; i++) {

                __m256d vb = _mm256_set_pd(0.0,
                                           blk[2],
                                           blk[1],
                                           blk[0]);

                __m256d prod = _mm256_mul_pd(vb, vx);

                // horizontal add
                __m128d low = _mm256_castpd256_pd128(prod);
                __m128d high = _mm256_extractf128_pd(prod, 1);
                __m128d sum2 = _mm_add_pd(low, high);
                __m128d sum1 = _mm_hadd_pd(sum2, sum2);

                double acc = _mm_cvtsd_f64(sum1);

                yrow[0] += acc;


                vb = _mm256_set_pd(0.0,
                                           blk[5],
                                           blk[4],
                                           blk[3]);

                prod = _mm256_mul_pd(vb, vx);

                // horizontal add
                low = _mm256_castpd256_pd128(prod);
                high = _mm256_extractf128_pd(prod, 1);
                sum2 = _mm_add_pd(low, high);
                sum1 = _mm_hadd_pd(sum2, sum2);

                acc = _mm_cvtsd_f64(sum1);

                yrow[1] += acc;




                vb = _mm256_set_pd(0.0,
                                           blk[8],
                                           blk[7],
                                           blk[6]);

                prod = _mm256_mul_pd(vb, vx);

                // horizontal add
                low = _mm256_castpd256_pd128(prod);
                high = _mm256_extractf128_pd(prod, 1);
                sum2 = _mm_add_pd(low, high);
                sum1 = _mm_hadd_pd(sum2, sum2);

                acc = _mm_cvtsd_f64(sum1);

                yrow[2] += acc;
            //}
        }
    }
}



void bc_matvec_avx3_best(const BlockedCSR *A, const double *x, double *y) {
    int bs = A->bs;           // = 3 neste caso
    int Nblocks = A->nb;
    int total_len = Nblocks * bs;

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];

        __m256d y0 = _mm256_set1_pd(0.0);
        __m256d y1 = _mm256_set1_pd(0.0);
        __m256d y2 = _mm256_set1_pd(0.0);

        for (int p = row_start; p < row_end; p++) {

            int bcol = A->ja[p];
            const double *blk  = &A->vals[(size_t)p * bs * bs];
            const double *xcol = &x[(size_t)bcol * bs];

            __m256d vx = _mm256_set_pd(0.0, xcol[2], xcol[1], xcol[0]);

            
            //for (int i = 0; i < bs; i++) {
                
                __m256d vb0 = _mm256_set_pd(0.0,
                                           blk[2],
                                           blk[1],
                                           blk[0]);

                y0 = _mm256_fmadd_pd(vb0, vx, y0 );

                __m256d  vb1 = _mm256_set_pd(0.0,
                                           blk[5],
                                           blk[4],
                                           blk[3]);

                y1 = _mm256_fmadd_pd(vb1, vx, y1);

                __m256d  vb2  = _mm256_set_pd(0.0,
                                           blk[8],
                                           blk[7],
                                           blk[6]);

                y2 = _mm256_fmadd_pd(vb2, vx, y2);
            //}
        }
        //yrow[0] = 1.0;
        yrow[0] = hadd_256( y0 );
        yrow[1] = hadd_256( y1 );
        yrow[2] = hadd_256( y2 );
    }
    
}



void bc_matvec_avx3(const BlockedCSR * restrict A, const double * restrict x, double * restrict y) {
    int bs = A->bs;           // = 3 neste caso
    int Nblocks = A->nb;
    int total_len = Nblocks * bs;
   
    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end   = A->ia[brow+1];

        double *yrow = &y[(size_t)brow * bs];
        int p = row_start;
        int bcol = A->ja[p];
        const double *blk  = &A->vals[(size_t)p * bs * bs];
        const double *xcol = &x[(size_t)bcol * bs];

        //_mm_prefetch((const char*)&blk[9], _MM_HINT_T0);
        //_mm_prefetch((const char*)&x[(size_t)A->ja[p+1] * bs], _MM_HINT_T0);
        


        __m256d vx = _mm256_loadu_pd( xcol );
        //for (int i = 0; i < bs; i++) {
            
        __m256d vb0 = _mm256_loadu_pd( blk);
        /*__m256d vb0 = _mm256_set_pd(blk[0],
                                    blk[1],
                                    blk[2], 
                                    blk[3] );
        */
        __m256d y0 = _mm256_mul_pd(vb0, vx );

        

        __m256d vb1 = _mm256_loadu_pd( &blk[3] );
        /*__m256d vb1 = _mm256_set_pd(blk[3],
                                    blk[4],
                                    blk[5], 
                                    blk[6] );
        */
        __m256d y1 = _mm256_mul_pd(vb1, vx );

        __m256d vb2  = _mm256_loadu_pd( &blk[6] );
        /*__m256d vb2 = _mm256_set_pd(blk[6],
                                    blk[7],
                                    blk[8], 
                                    blk[9] );
        */

        __m256d y2 = _mm256_mul_pd(vb2, vx );

        for (int p = row_start + 1; p < row_end; p++) {
            bcol = A->ja[p];
            blk  = &A->vals[(size_t)p * bs * bs];
            xcol = &x[(size_t)bcol * bs];
            __m256d vx = _mm256_loadu_pd( xcol );
            
            vb0 = _mm256_loadu_pd( blk);
            y0  = _mm256_fmadd_pd(vb0, vx, y0 );
            
            vb1 = _mm256_loadu_pd( &blk[3] );
            y1 = _mm256_fmadd_pd(vb1, vx, y1);
            
            vb2  = _mm256_loadu_pd( &blk[6] );
            y2 = _mm256_fmadd_pd(vb2, vx, y2);
        
        }
        yrow[0] = hadd_256( y0 );
        yrow[1] = hadd_256( y1 );
        yrow[2] = hadd_256( y2 );   
    }
}

int main(int argc, char **argv) {

    if (argc != 5) {
        printf("%s <inicio> <fim> <incremento> <K>\n", argv[0]);
        return 1;
    }

    int ini  = atoi(argv[1]);
    int fim  = atoi(argv[2]);
    int inc  = atoi(argv[3]);
    int K    = atoi(argv[4]);

    if (ini <= 0 || fim < ini || inc <= 0 || K <= 0) {
        printf("Parâmetros inválidos.\n");
        return 1;
    }

    // Loop principal — nx = ny = nz
    for (int nx = ini; nx <= fim; nx += inc) {
        int ny = nx;
        int nz = nx;
        int N = nx * ny * nz;

        // Gera matriz e vetores
        BlockedCSR *A = generate_blocked27_3x3(nx, ny, nz);

        double *x     = malloc((size_t)3*N * sizeof(double));
        double *y_ref = malloc((size_t)3*N * sizeof(double));
        double *y_avx = malloc((size_t)3*N * sizeof(double));

        for (int i = 0; i < 3*N; i++)
            x[i] = i;

        // ============================
        //   Tempo escalar
        // ============================
        double t0 = wtime();
        for (int k = 0; k < K; k++)
            bc_matvec(A, x, y_ref);
        double t1 = wtime();

        double time_scalar = (t1 - t0) / K;

        // ============================
        //   Tempo AVX
        // ============================
        double t2 = wtime();
        for (int k = 0; k < K; k++)
            bc_matvec_avx3(A, x, y_avx);
        double t3 = wtime();

        double time_avx = (t3 - t2) / K;

        // ============================
        //   Verificação de erro
        // ============================
        double max_err = 0.0;
        for (int i = 0; i < 3*N; i++) {
            double diff = fabs(y_ref[i] - y_avx[i])/fabs( y_ref[i] );
            if (diff > max_err) max_err = diff;
        }

        //printf("MAX ERROR %.6g\n", max_err);
        // ============================
        //   Saída resumida para plot
        // ============================
        // printf("%d & %.6f\\\\\n", 3*N, time_scalar / time_avx);
        printf("Tempo escalar: %.6f s | Tempo AVX: %.6f s | Speedup: %.2fx | Melhora: %.2f%% | N = %d\n",
               time_scalar, time_avx, time_scalar / time_avx, (time_scalar - time_avx) / time_avx * 100, 3*N);

        // limpa
        free(x);
        free(y_ref);
        free(y_avx);
        bc_free(A);
    }

    return 0;
}




// int main2(int argc, char **argv) {
//     int nx, ny, nz;

//     if (argc != 4) {
//         printf("%s <nx> <ny> <nz>\n", argv[0]);
//         return 1;
//     }

//     nx = atoi(argv[1]);
//     ny = atoi(argv[2]);
//     nz = atoi(argv[3]);

//     int N = nx * ny * nz;
//     printf("Gerando blocked CSR 27-pt stencil para malha %dx%dx%d (N=%d)\n",
//            nx, ny, nz, N);

//     BlockedCSR *A = generate_blocked27_3x3(nx, ny, nz);

//     double *x      = malloc((size_t)3*N * sizeof(double));
//     double *y_ref  = malloc((size_t)3*N * sizeof(double));
//     double *y_avx  = malloc((size_t)3*N * sizeof(double));

//     for (int i = 0; i < 3*N; i++) x[i] = 1.0;

//     int K = 100;  // número de repetições

//     // ============================
//     //   Tempo escalar
//     // ============================
//     double t0 = wtime();
//     for (int k = 0; k < K; k++)
//         bc_matvec(A, x, y_ref);
//     double t1 = wtime();

//     double time_scalar = (t1 - t0) / K;


//     // ============================
//     //   Tempo AVX
//     // ============================
//     double t2 = wtime();
//     for (int k = 0; k < K; k++)
//         bc_matvec_avx3(A, x, y_avx);
//     double t3 = wtime();

//     double time_avx = (t3 - t2) / K;

//     // ============================
//     //   Verificação
//     // ============================
//     double max_err = 0.0;
//     for (int i = 0; i < 3*N; i++) {
//         double diff = fabs(y_ref[i] - y_avx[i]);
//         if (diff > max_err) max_err = diff;
//     }

//     printf("\n--- Resultados ---\n");
//     printf("Tempo escalar = %.6f s\n", time_scalar);
//     printf("Tempo AVX     = %.6f s\n", time_avx);
//     printf("Speedup       = %.2f x\n", time_scalar / time_avx);
//     printf("Máximo erro   = %.6e\n\n", max_err);

//     free(x);
//     free(y_ref);
//     free(y_avx);
//     bc_free(A);
//     return 0;
// }
