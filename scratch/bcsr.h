#ifndef BCSR_H
#define BCSR_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define BS 3

#define idx(i, j) i * BS + j

struct BlockedCSR {
   int nb;       // número de "block rows" (nós)
   int bs;       // block size vamos usar 3 
   int nnzb;     // número de blocos não nulos
   int *ia;      // tamanho nb+1, índice inicial de cada block-row em ja/vals
   int *ja;      // tamanho nnzb, coluna (block index) de cada bloco
   double *vals; // tamanho nnzb * bs * bs, blocos armazenados consecutivamente em row-major dentro do bloco
};


BlockedCSR *bc_alloc(int nb, int bs, int max_nblocks) {
    BlockedCSR *A = (BlockedCSR *) malloc(sizeof(BlockedCSR));

    if (!A) {
        perror("malloc A");
        exit(1);
    }

    A->nb = nb;
    A->bs = bs;
    A->nnzb = 0;
    A->ia   = (int *) malloc((nb + 1) * sizeof(int));
    A->ja   = (int *) malloc(max_nblocks * sizeof(int));
    A->vals = (double *) malloc(max_nblocks * bs * bs * sizeof(double));

    if (!A->ia || !A->ja || !A->vals) {
        perror("malloc arrays");
        exit(1);
    }

    A->ia[0] = 0;
    return A;
}

void bc_shrink_to_fit(BlockedCSR *A) {
    A->ja   = (int *) realloc(A->ja, A->nnzb * sizeof(int));
    A->vals = (double *) realloc(A->vals, A->nnzb * A->bs * A->bs * sizeof(double));
}

void bc_free(BlockedCSR *A) {
    if (!A) {
        return;
    }

    free(A->ia);
    free(A->ja);
    free(A->vals);
    free(A);
}

void bc_push_block(BlockedCSR *A, int brow, int bcol, const double *block) {
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

void bc_draw(const BlockedCSR *A) {
    int nb = A->nb;

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < nb; j++) {
            bool is_block = false;
            int row_start = A->ia[i];
            int row_end   = A->ia[i + 1];

            for(int idx = row_start; idx < row_end; idx++) {
                if(j == A->ja[idx]) {
                    printf("[X]");
                    is_block = true;
                    break;
                }           
            }

            if(!is_block) {
                printf("   ");
            } 
        }

        printf("\n");
    }
}

double *get_block(const BlockedCSR *A, int row, int col) {
    int row_start = A->ia[row];
    int row_end   = A->ia[row + 1];

    for (int p = row_start; p < row_end; p++) {
        if (A->ja[p] == col) {
            return &A->vals[(size_t)p * A->bs * A->bs];
        }
    }

    return nullptr;
}


void draw_global_matrix(const BlockedCSR *A) {
    int nb = A->nb;
    int bs = A->bs;

    for (int i = 0; i < nb; i++) {
        for (int row = 0; row < bs; row++) {
            for (int j = 0; j < nb; j++) {
                double *block = get_block(A, i, j);

                for (int col = 0; col < bs; col++) {
                    if (block != nullptr) {
                        printf("%.2f ", block[row * bs + col]);
                    } else {
                        printf("  ");
                    }
                }
            }
            printf("\n");
        }
    }
}

#endif // BCSR_H