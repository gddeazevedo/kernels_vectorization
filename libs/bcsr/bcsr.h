#pragma once

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

class BlockCSR
{
   public:
      int nb;        // número de "block rows" (nós). number of rows
      int bs;        // block size vamos usar 3. block size
      int nnzb;      // número de blocos não nulos. nonzero blocks
      int *brptr;    // tamanho nb+1, índice inicial de cada block-row em bcind/bvals. block rows pointers
      int *bcind;    // tamanho nnzb, coluna (block index) de cada bloco. block column indexes
      double *bvals; // tamanho nnzb * bs * bs, blocos armazenados consecutivamente em row-major dentro do bloco. block values

      BlockCSR(int nb, int bs, int max_nblocks);
      ~BlockCSR();

      BlockCSR(const BlockCSR &) = delete;
      BlockCSR &operator=(const BlockCSR &) = delete;

      BlockCSR(BlockCSR &&other) noexcept;
      BlockCSR &operator=(BlockCSR &&other) noexcept;

      void shrink_to_fit();
      void push_block(const int row, const int col, const double *block);
      void draw() const;
      double *get_block(const int row, const int col);

      static BlockCSR generate_blocked27_3x3(int nx, int ny, int nz);
};