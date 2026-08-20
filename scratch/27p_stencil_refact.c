#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define BLOCK_SIZE      3
#define EXPECTED_INPUTS 4

typedef struct {
    int    n_block_rows;
    int    block_size;
    int    n_nonzero_blocks;
    int    *row_blocks_ptrs;
    int    *col_blocks_indices;
    double *block_values;
} BlockedCSR;


void bc_draw(BlockedCSR *A);
void bc_shrink_to_fit(BlockedCSR *A);
void bc_free(BlockedCSR *A);
void bc_push_block(BlockedCSR *A, int block_row, int block_col, const double *block);
void bc_matvec(const BlockedCSR *A, const double *input, double *output);
BlockedCSR *bc_alloc(int n_block_rows, int block_size, int max_n_blocks);
BlockedCSR *create_blocked_27pt_3x3(int nx, int ny, int nz);
unsigned char is_not_neighbour(int x, int y, int z, int nx, int ny, int nz);


int main(int argc, char **argv) {
    if (argc != EXPECTED_INPUTS) {
        printf("Usage: %s <nx> <ny> <nz>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);

    int n_nodes = nx * ny * nz;
    printf("Generating blocked CSR 27-pt stencil for mesh %dx%dx%d (N=%d)\n", nx, ny, nz, n_nodes);

    BlockedCSR *A = create_blocked_27pt_3x3(nx, ny, nz);

    bc_draw(A);

    double *input  = calloc((size_t)BLOCK_SIZE * n_nodes, sizeof(double));
    double *output = calloc((size_t)BLOCK_SIZE * n_nodes, sizeof(double));

    if (!input || !output) {
        perror("calloc input/output");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < BLOCK_SIZE * n_nodes; i++) {
        input[i] = 1.0;
    }

    // bc_matvec(A, input, output);
    // printf("\nResult of matvec:\n");
    // for (int i = 0; i < BLOCK_SIZE * n_nodes; i++) {
    //     printf("output[%d] = %f\n", i, output[i]);
    // }

    free(input);
    free(output);
    bc_free(A);
    return EXIT_SUCCESS;
}

void bc_draw(BlockedCSR *A) {
    int n_nodes = A->n_block_rows;

    for (int i = 0; i < n_nodes; i++) {
        for (int j = 0; j < n_nodes; j++) {
            unsigned char is_block = 0;

            int row_start = A->row_blocks_ptrs[i];
            int row_end   = A->row_blocks_ptrs[i + 1];

            for (int k = row_start; k < row_end; k++) {
                if (j == A->col_blocks_indices[k]) {
                    printf("[X]");
                    is_block = 1;
                    break;
                }
            }

            if (!is_block) {
                printf("   ");
            }
        }

        printf("\n");
    }
}

void bc_shrink_to_fit(BlockedCSR *A) {
    A->col_blocks_indices = realloc(A->col_blocks_indices, (size_t)A->n_nonzero_blocks * sizeof(int));
    A->block_values = realloc(A->block_values, (size_t)A->n_nonzero_blocks * A->block_size * A->block_size * sizeof(double));
}

void bc_free(BlockedCSR *A) {
    if (A == NULL) {
        return;
    }

    free(A->row_blocks_ptrs);
    free(A->col_blocks_indices);
    free(A->block_values);
    free(A);
}

void bc_push_block(BlockedCSR *A, int block_row, int block_col, const double *block) {
    int block_size = A->block_size;
    int pos = A->n_nonzero_blocks;
    A->col_blocks_indices[pos] = block_col;
    memcpy(&A->block_values[(size_t)pos * block_size * block_size], block, (size_t)block_size * block_size * sizeof(double));
    A->n_nonzero_blocks++;
    A->row_blocks_ptrs[block_row + 1] = A->n_nonzero_blocks;
}

void bc_matvec(const BlockedCSR *A, const double *input, double *output) {
    int n_block_rows = A->n_block_rows;
    int block_size = A->block_size;

    int total_len = n_block_rows * block_size;

    for (int i = 0; i < total_len; i++) {
        output[i] = 0.0;
    }

    for (int block_row = 0; block_row < n_block_rows; block_row++) {
        int row_start = A->row_blocks_ptrs[block_row];
        int row_end   = A->row_blocks_ptrs[block_row + 1];
        double *output_row = &output[(size_t)block_row * block_size];

        for (int block = row_start; block < row_end; block++) {
            int block_col = A->col_blocks_indices[block];
            const double *block_values = &A->block_values[(size_t)block * block_size * block_size];
            const double *input_col = &input[(size_t)block_col * block_size];

            for (int i = 0; i < block_size; i++) {
                double acc = 0.0;

                // ordenamento natural: row major
                acc += block_values[i * block_size] * input_col[0];
                acc += block_values[i * block_size + 1] * input_col[1];
                acc += block_values[i * block_size + 2] * input_col[2];
                output_row[i] += acc;
            }
        }
    }
}

BlockedCSR *bc_alloc(int n_block_rows, int block_size, int n_blocks) {
    BlockedCSR *A = malloc(sizeof(BlockedCSR));

    if (A == NULL) {
        perror("malloc A");
        exit(EXIT_FAILURE);
    }

    A->n_block_rows     = n_block_rows;
    A->block_size       = block_size;
    A->n_nonzero_blocks = 0;

    A->row_blocks_ptrs    = malloc((n_block_rows + 1) * sizeof(int));
    A->col_blocks_indices = malloc(n_blocks * sizeof(int));
    A->block_values       = malloc((size_t)n_blocks * block_size * block_size * sizeof(double));

    A->row_blocks_ptrs[0] = 0; // The first row block starts at index 0 in block_values

    return A;
}

BlockedCSR *create_blocked_27pt_3x3(int nx, int ny, int nz) {
    int n_nodes = nx * ny * nz;
    int n_blocks_max = 27 * n_nodes; // Overestimate for maximum number of blocks

    BlockedCSR *A = bc_alloc(n_nodes, BLOCK_SIZE, n_blocks_max);

    unsigned int n_nz_count = 0;

    for (int iz = 0; iz < nz; iz++) {
        for (int iy = 0; iy < ny; iy++) {
            for (int ix = 0; ix < nx; ix++) {
                int node_index = ix + iy * nx + iz * nx * ny;
                A->row_blocks_ptrs[node_index] = n_nz_count;

                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int neighbor_x = ix + dx;
                            int neighbor_y = iy + dy;
                            int neighbor_z = iz + dz;

                            if (is_not_neighbour(neighbor_x, neighbor_y, neighbor_z, nx, ny, nz)) {
                                continue;
                            }

                            int neighbor_index = neighbor_x + neighbor_y * nx + neighbor_z * nx * ny;
                            double block[BLOCK_SIZE * BLOCK_SIZE] = {0};

                            if (neighbor_index == node_index) {
                                // diagonal do nó
                                block[0] = 2.0;
                                block[4] = 2.0;
                                block[8] = 2.5;

                                // pequenos acoplamentos
                                block[1] = block[2] = block[3] = block[5] = block[6] = block[7] = 0.1;
                            } else {
                                // vizinho: valores menores
                                for (int r = 0; r < BLOCK_SIZE * BLOCK_SIZE; r++) {
                                    block[r] = 0.05;
                                }
                            }

                            bc_push_block(A, node_index, neighbor_index, block);
                            n_nz_count++;
                        }
                    }
                }
            }
        }
    }

    A->n_nonzero_blocks = n_nz_count;
    bc_shrink_to_fit(A);
    return A;
}

// Check if the given (x, y, z) coordinates are out of bounds
// nx - number of nodes in x direction
// ny - number of nodes in y direction
// nz - number of nodes in z direction
unsigned char is_not_neighbour(int x, int y, int z, int nx, int ny, int nz) {
    return (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz);
}
