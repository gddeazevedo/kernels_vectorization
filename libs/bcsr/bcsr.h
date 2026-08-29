#pragma once

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

class BlockedCSR
{
   public:
      int nb;        // número de "block rows" (nós). number of rows
      int bs;        // block size vamos usar 3. block size
      int nnzb;      // número de blocos não nulos. nonzero blocks
      int *brptr;    // tamanho nb+1, índice inicial de cada block-row em bcind/bvals. block rows pointers
      int *bcind;    // tamanho nnzb, coluna (block index) de cada bloco. block column indexes
      double *bvals; // tamanho nnzb * bs * bs, blocos armazenados consecutivamente em row-major dentro do bloco. block values

      BlockedCSR(int nb, int bs, int max_nblocks);
      ~BlockedCSR();

      BlockedCSR(const BlockedCSR &) = delete;
      BlockedCSR &operator=(const BlockedCSR &) = delete;

      BlockedCSR(BlockedCSR &&other) noexcept;
      BlockedCSR &operator=(BlockedCSR &&other) noexcept;

      void shrink_to_fit();
      void push_block(const int row, const int col, const double *block);
      void draw() const;
      double *get_block(const int row, const int col);

      static BlockedCSR generate_blocked27_3x3(int nx, int ny, int nz);
};