#include <bcsr.h>

BlockedCSR::BlockedCSR(int nb, int bs, int max_nblocks) : nb(nb), bs(bs), nnzb(0)
{
    this->ia   = (int *) malloc((nb + 1) * sizeof(int));
    this->ja   = (int *) malloc(max_nblocks * sizeof(int));
    this->vals = (double *) malloc(max_nblocks * bs * bs * sizeof(double));

    if (!this->ia || !this->ja || !this->vals) {
        perror("malloc arrays");
        exit(1);
    }

    this->ia[0] = 0;
}

BlockedCSR::~BlockedCSR() {
    free(this->ia);
    free(this->ja);
    free(this->vals);
}

BlockedCSR::BlockedCSR(BlockedCSR &&other) noexcept
    : nb(other.nb), bs(other.bs), nnzb(other.nnzb),
      ia(other.ia), ja(other.ja), vals(other.vals)
{
    other.ia   = nullptr;
    other.ja   = nullptr;
    other.vals = nullptr;
}

BlockedCSR &BlockedCSR::operator=(BlockedCSR &&other) noexcept {
    if (this != &other) {
        free(this->ia);
        free(this->ja);
        free(this->vals);

        this->nb = other.nb;
        this->bs = other.bs;
        this->nnzb = other.nnzb;
        this->ia = other.ia;
        this->ja = other.ja;
        this->vals = other.vals;

        other.ia = nullptr;
        other.ja = nullptr;
        other.vals = nullptr;
    }
    return *this;
}

void BlockedCSR::shrink_to_fit() {
    this->ja   = (int *) realloc(this->ja, this->nnzb * sizeof(int));
    this->vals = (double *) realloc(this->vals, this->nnzb * this->bs * this->bs * sizeof(double));
}

void BlockedCSR::push_block(int row, int col, const double *block) {
    int pos = this->nnzb;
    this->ja[pos] = col;
    memcpy(&this->vals[(size_t)pos * this->bs * this->bs], block, (size_t)this->bs * this->bs * sizeof(double));
    this->nnzb++;
    this->ia[row + 1] = this->nnzb;
}

BlockedCSR BlockedCSR::generate_blocked27_3x3(int nx, int ny, int nz) {
    int N = nx * ny * nz;
    int bs = 3;
    int max_blocks = N * 27;
    BlockedCSR A(N, bs, max_blocks);

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

void BlockedCSR::draw() const {
    for(int i = 0; i < this->nb; i++) {
        for(int j = 0; j < this->nb; j++) {
            bool is_block = false;
            int row_start = this->ia[i];
            int row_end   = this->ia[i + 1];

            for(int idx = row_start; idx < row_end; idx++) {
                if(j == this->ja[idx]) {
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

double *BlockedCSR::get_block(const int row, const int col) {
    int row_start = this->ia[row];
    int row_end   = this->ia[row + 1];
    int bs2 = this->bs * this->bs;

    for (int p = row_start; p < row_end; p++) {
        if (ja[p] == col) {
            return &this->vals[(size_t)p * bs2];
        }
    }

    return nullptr;
}