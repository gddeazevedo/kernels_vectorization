'''
1 2 3 | 0 0 0    | 10 11 12
4 5 6 | 0 0 0    | 13 14 15
7 8 9 | 0 0 0    | 16 17 18
------------------------
0 0 0 | 0 0 0    | 10 11 12
0 0 0 | 0 0 0    | 13 14 15
0 0 0 | 0 0 0    | 16 17 18
------------------------
1 2 3 | -1 -2 -3 | 0 0 0
4 5 6 | -3 -5 -6 | 0 0 0
7 8 9 | -7 -8 -9 | 0 0 0
------------------------
1 2 0 | 0 0 0    | 0 0 0
4 0 6 | 0 0 0    | 0 0 0
0 8 9 | 0 0 0    | 0 0 0
'''
import numpy as np
from numpy.typing import NDArray


class BlockedCSR2D:
    '''
    Comprime um tensor de duas dimensões
    '''
    def __init__(self, block_size: int):
        self.row_blocks_ptrs = [0]
        self.col_blocks_indices = []
        self.block_values = []
        self.block_size = block_size
        self.n_nz_blocks = 0

    def compress_matrix(self, A: NDArray) -> None:
        nx = A.shape[0]
        ny = A.shape[1]
        n_nonzero_blocks = 0
        for row in range(0, nx, self.block_size):
            for col in range(0, ny, self.block_size):
                block = self.build_block(A, row, col)

                if len(block) == 0:
                    continue

                n_nonzero_blocks += 1
                self.block_values.extend(block)
                self.col_blocks_indices.append(col // self.block_size)

            self.row_blocks_ptrs.append(n_nonzero_blocks)

        self.n_nz_blocks = n_nonzero_blocks

    def build_block(self, A: NDArray, row: int, col: int) -> list:
        block = []
        for i in range(self.block_size):
            for j in range(self.block_size):
                block.append(int(A[row + i][col + j]))

        if sum(block) != 0:
            return block

        return []

    def get_block_values(self, i: int):
        if i >= self.n_nz_blocks or i < 0:
            print("Out of bounds")
            return []
        block_start = i * self.block_size**2
        block_end   = (i + 1) * self.block_size**2
        return self.block_values[block_start : block_end]

    @property
    def n_block_rows(self):
        return len(self.row_blocks_ptrs) - 1

    def __str__(self) -> str:
        obj_repr = ""
        for i in range(self.n_block_rows):
            row_start = self.row_blocks_ptrs[i]
            row_end   = self.row_blocks_ptrs[i + 1]

            for block in range(row_start, row_end):
                block_col = self.col_blocks_indices[block]
                block_values = self.get_block_values(block)
                obj_repr += f"Bloco {block}: Linha {i} | Coluna {block_col} | {block_values}\n"
        return obj_repr


def bcsr_matvec(bcsr: BlockedCSR2D, x: list):
    block_size = bcsr.block_size
    b = [0] * (bcsr.n_block_rows * block_size)

    for line in range(bcsr.n_block_rows):
        row_start = bcsr.row_blocks_ptrs[line]
        row_end   = bcsr.row_blocks_ptrs[line + 1]

        for block in range(row_start,row_end):
            block_col = bcsr.col_blocks_indices[block]
            block_values = bcsr.get_block_values(block)
            x_slice = x[block_col * block_size : (block_col + 1) * block_size]

            for i in range(block_size):
                block_values_slice = block_values[i * block_size : (i + 1) * block_size]
                b[line * block_size + i] += np.dot(block_values_slice, x_slice)

    return b


A = np.array([
    [1, 2, 3,  0, 0, 0, 10, 11, 12],
    [4, 5, 6,  0, 0, 0, 13, 14, 15],
    [7, 8, 9,  0, 0, 0, 16, 17, 18],
    [0, 0, 0,  0, 0, 0, 10, 11, 12],
    [0, 0, 0,  0, 0, 0, 13, 14, 15],
    [0, 0, 0,  0, 0, 0, 16, 17, 18],
    [1, 2, 3, -1, -2, -3, 0, 0, 0],
    [4, 5, 6, -3, -5, -6, 0, 0, 0],
    [7, 8, 9, -7, -8, -9, 0, 0, 0],
    [1, 2, 0,  0, 0, 0, 0, 0, 0],
    [4, 0, 6,  0, 0, 0, 0, 0, 0],
    [0, 8, 9,  0, 0, 0, 0, 0, 0],
])

block_size = 3
bcsr = BlockedCSR2D(block_size)
bcsr.compress_matrix(A)
print(bcsr)
print(f"Rows ptrs: {bcsr.row_blocks_ptrs}\nBlock col: {bcsr.col_blocks_indices}", end="\n\n")


'''
Calculate Ax = b
Matrix dimension m x n
x dimentsion n x 1
b dimension m x 1
'''
x = [1] * block_size * block_size
# print(x)
b = bcsr_matvec(bcsr, x)

print(f"Matvec: {b}\n")

b = np.dot(A, x)

print(f"Esperado: {b}")


print(bcsr.n_block_rows)
print(bcsr.n_nz_blocks)
print(len(bcsr.block_values) == bcsr.n_nz_blocks * block_size**2)
