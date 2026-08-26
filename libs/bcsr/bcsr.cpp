#include <bcsr.h>

BlockedCSR::BlockedCSR(int nb, int bs, int max_nblocks)
{
    this->nnzb  = 0;
    this->nb    = nb;
    this->bs    = bs;
    this->brptr = (int *) malloc((nb + 1) * sizeof(int));
    this->bcind = (int *) malloc(max_nblocks * sizeof(int));
    this->bvals = (double *) malloc(max_nblocks * bs * bs * sizeof(double));

    if (!this->brptr || !this->bcind || !this->bvals) {
        perror("malloc arrays");
        exit(1);
    }

    this->brptr[0] = 0;
}

BlockedCSR::~BlockedCSR()
{
    free(this->brptr);
    free(this->bcind);
    free(this->bvals);
}

BlockedCSR::BlockedCSR(BlockedCSR &&other) noexcept
    : nb(other.nb), bs(other.bs), nnzb(other.nnzb),
      brptr(other.brptr), bcind(other.bcind), bvals(other.bvals)
{
    other.brptr = nullptr;
    other.bcind = nullptr;
    other.bvals = nullptr;
}

BlockedCSR &BlockedCSR::operator=(BlockedCSR &&other) noexcept
{
    if (this != &other) {
        free(this->brptr);
        free(this->bcind);
        free(this->bvals);

        this->nb = other.nb;
        this->bs = other.bs;
        this->nnzb = other.nnzb;
        this->brptr = other.brptr;
        this->bcind = other.bcind;
        this->bvals = other.bvals;

        other.brptr = nullptr;
        other.bcind = nullptr;
        other.bvals = nullptr;
    }
    return *this;
}

void BlockedCSR::shrink_to_fit()
{
    this->bcind = (int *) realloc(this->bcind, this->nnzb * sizeof(int));
    this->bvals = (double *) realloc(this->bvals, this->nnzb * this->bs * this->bs * sizeof(double));
}

void BlockedCSR::push_block(int row, int col, const double *block)
{
    int pos = this->nnzb;
    this->bcind[pos] = col;
    memcpy(&this->bvals[(size_t)pos * this->bs * this->bs], block, (size_t)this->bs * this->bs * sizeof(double));
    this->nnzb++;
    this->brptr[row + 1] = this->nnzb;
}

void BlockedCSR::draw() const
{
    for(int i = 0; i < this->nb; i++) {
        for(int j = 0; j < this->nb; j++) {
            bool is_block = false;
            int row_start = this->brptr[i];
            int row_end   = this->brptr[i + 1];

            for(int idx = row_start; idx < row_end; idx++) {
                if(j == this->bcind[idx]) {
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

double *BlockedCSR::get_block(const int row, const int col)
{
    int row_start = this->brptr[row];
    int row_end   = this->brptr[row + 1];

    while (row_start < row_end) {
        int mid = (row_start + row_end) >> 1;

        if (this->bcind[mid] < col) {
            row_start = mid + 1;
        } else if (this->bcind[mid] > col) {
            row_end = mid;
        } else {
            return &this->bvals[(size_t) mid * this->bs * this->bs];
        }
    }

    return nullptr;
}

BlockedCSR BlockedCSR::generate_blocked27_3x3(int nx, int ny, int nz)
{
    int N = nx * ny * nz;
    int bs = 3;
    int max_blocks = N * 27;
    BlockedCSR A(N, bs, max_blocks);

    int nnz_count = 0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int id = i + j * nx + k * nx * ny;
                A.brptr[id] = nnz_count;

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

                A.brptr[id + 1] = nnz_count;
            }
        }
    }

    A.nnzb = nnz_count;
    A.shrink_to_fit();
    return A;
}