#ifndef BCSR_CLASS_H
#define BCSR_CLASS_H

#include <cstdlib>
#include <cstdio>
#include <cstring>

#define BS 3
#define idx(i, j) ((i) * BS + (j))

class BlockedCSR {
public:
    int nb;
    int bs;
    int nnzb;
    int *ia;
    int *ja;
    double *vals;

    BlockedCSR(int nb, int bs, int max_nblocks)
        : nb(nb), bs(bs), nnzb(0)
    {
        ia   = (int *) malloc((nb + 1) * sizeof(int));
        ja   = (int *) malloc(max_nblocks * sizeof(int));
        vals = (double *) malloc((size_t)max_nblocks * bs * bs * sizeof(double));

        if (!ia || !ja || !vals) {
            perror("BlockedCSR malloc");
            exit(1);
        }

        ia[0] = 0;
    }

    ~BlockedCSR() {
        free(ia);
        free(ja);
        free(vals);
    }

    // Sem cópia (evita double-free)
    BlockedCSR(const BlockedCSR &) = delete;
    BlockedCSR &operator=(const BlockedCSR &) = delete;

    // Move OK
    BlockedCSR(BlockedCSR &&other) noexcept
        : nb(other.nb), bs(other.bs), nnzb(other.nnzb),
          ia(other.ia), ja(other.ja), vals(other.vals)
    {
        other.ia   = nullptr;
        other.ja   = nullptr;
        other.vals = nullptr;
    }

    BlockedCSR &operator=(BlockedCSR &&other) noexcept {
        if (this != &other) {
            free(ia); free(ja); free(vals);
            nb = other.nb; bs = other.bs; nnzb = other.nnzb;
            ia = other.ia; ja = other.ja; vals = other.vals;
            other.ia = nullptr; other.ja = nullptr; other.vals = nullptr;
        }
        return *this;
    }

    void push_block(int brow, int bcol, const double *block) {
        int pos = nnzb;
        ja[pos] = bcol;
        memcpy(&vals[(size_t)pos * bs * bs], block, (size_t)bs * bs * sizeof(double));
        nnzb++;
        ia[brow + 1] = nnzb;
    }

    void shrink_to_fit() {
        ja   = (int *) realloc(ja, nnzb * sizeof(int));
        vals = (double *) realloc(vals, (size_t)nnzb * bs * bs * sizeof(double));
    }

    double *get_block(int row, int col) const {
        int row_start = ia[row];
        int row_end   = ia[row + 1];

        for (int p = row_start; p < row_end; p++) {
            if (ja[p] == col) {
                return &vals[(size_t)p * bs * bs];
            }
        }

        return nullptr;
    }

    void draw() const {
        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < nb; j++) {
                bool found = false;

                for (int p = ia[i]; p < ia[i + 1]; p++) {
                    if (ja[p] == j) {
                        printf("[X]");
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    printf("   ");
                }
            }
            printf("\n");
        }
    }

    void draw_global() const {
        for (int i = 0; i < nb; i++) {
            for (int row = 0; row < bs; row++) {
                for (int j = 0; j < nb; j++) {
                    double *block = get_block(i, j);

                    for (int col = 0; col < bs; col++) {
                        if (block != nullptr) {
                            printf("%8.4f ", block[row * bs + col]);
                        } else {
                            printf("         ");
                        }
                    }
                }
                printf("\n");
            }
        }
    }

    static BlockedCSR generate_blocked27(int nx, int ny, int nz) {
        int N = nx * ny * nz;
        BlockedCSR A(N, BS, N * 27);

        int nnz_count = 0;

        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int id = i + j * nx + k * nx * ny;
                    A.ia[id] = nnz_count;

                    for (int dk = -1; dk <= 1; dk++) {
                        for (int dj = -1; dj <= 1; dj++) {
                            for (int di = -1; di <= 1; di++) {
                                int ni = i + di;
                                int nj = j + dj;
                                int nk = k + dk;

                                if (ni < 0 || nj < 0 || nk < 0 ||
                                    ni >= nx || nj >= ny || nk >= nz) {
                                    continue;
                                }

                                int nid = ni + nj * nx + nk * nx * ny;
                                double blk[9];

                                if (nid == id) {
                                    for (int r = 0; r < 9; r++) blk[r] = 0.0;
                                    blk[0] = 2.0; blk[4] = 2.0; blk[8] = 2.5;
                                    blk[1] = blk[2] = blk[3] = blk[5] = blk[6] = blk[7] = 0.1;
                                } else {
                                    for (int r = 0; r < 9; r++) blk[r] = 0.05;
                                }

                                A.push_block(id, nid, blk);
                                nnz_count++;
                            }
                        }
                    }

                    A.ia[id + 1] = nnz_count;
                }
            }
        }

        A.nnzb = nnz_count;
        A.shrink_to_fit();
        return A;
    }
};

#endif // BCSR_CLASS_H