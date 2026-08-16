#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int nb;      // número de "block rows" (nós)
    int bs;      // block size vamos usar 3 
    int nnzb;    // número de blocos não nulos
    int *ia;     // tamanho nb+1, índice inicial de cada block-row em ja/vals
    int *ja;     // tamanho nnzb, coluna (block index) de cada bloco
    double *vals;// tamanho nnzb * bs * bs, blocos armazenados consecutivamente em row-major dentro do bloco
} BlockedCSR;


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
    printf("MAX BLOCKS = %d\n", max_blocks);
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
                            if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) continue;  // borda da malha
                            int nid = ni + nj * nx + nk * nx * ny;
                            double blk[9];
                            if (nid == id) { // ponto -é ele mesmo, diagonal é o proprio ponto
                                // bloco diagonal
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

void bc_matvec(const BlockedCSR *A, const double *x, double *y) {
    int bs = A->bs;
    int Nblocks = A->nb;
    // zera y
    int total_len = Nblocks * bs;
    for (int i = 0; i < total_len; i++) y[i] = 0.0;

    for (int brow = 0; brow < A->nb; brow++) {
        int row_start = A->ia[brow];
        int row_end = A->ia[brow + 1];
        double *yrow = &y[(size_t)brow * bs];

        for (int p = row_start; p < row_end; p++) {
            int bcol = A->ja[p];
            const double *blk = &A->vals[(size_t)p * bs * bs]; // vetor de dimensão 3 x 3
            const double *xcol = &x[(size_t)bcol * bs]; // vetor de dimensão 3

            for (int i = 0; i < bs; i++) {
                double acc = 0.0;
                // unroll 3
                acc += blk[i*bs + 0] * xcol[0];
                acc += blk[i*bs + 1] * xcol[1];
                acc += blk[i*bs + 2] * xcol[2];
                yrow[i] += acc;
            }
        }
    }
}

int main(int argc, char **argv) {
    int nx, ny, nz;
    
    if( argc != 4 ) {
        printf("%s <nx> <ny> <nz>\n", argv[0] );
        return 1;
    }
    
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    
    int N = nx * ny * nz;
    printf("Gerando blocked CSR 27-pt stencil para malha %dx%dx%d (N=%d)\n", nx, ny, nz, N);

    BlockedCSR *A = generate_blocked27_3x3(nx, ny, nz);
    bc_draw( A );

    double *x = malloc((size_t)3 * N * sizeof(double));
    double *y = malloc((size_t)3 * N * sizeof(double));
    for (int i = 0; i < 3*N; i++) x[i] = 1.0;

    // matvec
    bc_matvec(A, x, y);

    free(x); free(y);
    bc_free(A);
    return EXIT_SUCCESS;
}

