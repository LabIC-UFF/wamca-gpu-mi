
#define GPU_GLOBAL_ALIGN 128
#define ADS_INFO_SIZE GPU_GLOBAL_ALIGN
#define ADS_INFO_ELEMS (ADS_INFO_SIZE / sizeof(uint))

/*!
 * MLADSInfo
 */
struct MLADSInfo {
  uint size;      ///< ADS buffer size in bytes
  uint rowElems;  ///< ADS row  elems
  uint solElems;  ///< Solution elems
  uint solCost;   ///< Solution cost
  uint tour;      ///< Tour (0/1)
  uint round;     ///< Round: 0=0.0, 1=0.5
};

/*!
 * MLADSData
 */
union MLADSData {
  MLADSInfo s;
  uint v[ADS_INFO_ELEMS];
};

/*!
 * MLMove64
 */
/*
struct MLMove64 {
    uint    id   :  4;
    uint    i    : 14;
    uint    j    : 14;
    int     cost : 32;
};
*/

struct MLMove64 {
  unsigned int id : 4;
  unsigned int i : 14;
  unsigned int j : 14;
  int cost;  // REMOVED :32 !! Goddamn Eyder...
};

/*!
 * MLMovePack
 */
union MLMovePack {
  MLMove64 s;
  ulong w;
  int i[2];
  uint u[2];
  long l[1];
};

__global__ void kernel2Opt(const MLADSData* gm_ads, MLMovePack* gm_move,
                           const int size) {
  /*!
   * Shared memory variables
   */
  extern __shared__ int sm_buffer[];  // Dynamic shared memory buffer
  /*!
   * Local variables
   */
  __shared__ int *sm_soldist,  // Client distances
      *sm_coordx,              // Clients x-coordinates
      *sm_coordy,              // Clients y-coordinates
      *sm_move,                // Thread movement cost
      sm_rsize,                // ADS row size
      sm_tour;                 // Tour/path

  uint* gm_data;  // Points to some ADS data
  int c,          // Chunk no
      ctx,        // Tx index for chunk
      cmax,       // Number of chunks
      csize;      // Chunk size
  int cost,       // Cost improvement
      i, j,       // Movement indexes
      k;          // Last solution index

  if (tx >= size) return;

  /*
   * Dynamic shared memory buffer usage
   *
   * buffer
   * |
   * v
   * +---------+--------+--------+------------------+
   * | soldist | coordx | coordy | movcost | moveid |
   * +---------+--------+--------+------------------+
   */

  /*
   * Set pointers to proper location in 'sm_buffer'
   * Only thread 0 initializes shared variables
   */
  if (tx == 0) {
    sm_soldist = sm_buffer;
    sm_coordx = sm_soldist + size;
    sm_coordy = sm_coordx + size;
    sm_move = sm_coordy + size;

    sm_rsize = gm_ads->s.rowElems;
    sm_tour = gm_ads->s.tour;
  }
  __syncthreads();

  // Number of chunks
  cmax = GPU_DIVCEIL(size, blockDim.x);
  // Chunk size
  csize = GPU_MIN(size, int(blockDim.x));

  // Get clients coordinates
  gm_data = ADS_COORD_PTR(gm_ads);
  for (c = 0; (c < cmax) && ((ctx = c * blockDim.x + tx) < size); c++) {
    sm_coordx[ctx] = GPU_HI_USHORT(gm_data[ctx]);
    sm_coordy[ctx] = GPU_LO_USHORT(gm_data[ctx]);
  }

  // Get solution distances
  gm_data = ADS_SOLUTION_PTR(gm_ads, sm_rsize);
  for (c = 0; (c < cmax) && ((ctx = c * blockDim.x + tx) < size); c++)
    sm_soldist[ctx] = GPU_LO_USHORT(gm_data[ctx]);

  // Initialize movement/cost
  sm_move[tx] = COST_INFTY;

  // Wait all threads arrange data
  __syncthreads();

  /*
      if(tx == 0 && by == 0) {
          kprintf("COORDS\n");
          for(i=0;i < size;i++)
              kprintf("%d\t(%d,%d)\n",i,sm_coordx[i],sm_coordy[i]);
          kprintf("\n");

          kprintf("DIST\n");
          for(i=0;i < size;i++)
              kprintf(" %d",sm_soldist[i]);
          kprintf("\n");
      }
  */

  // Get solution distances
  for (c = 0; c < cmax; c++) {
    ctx = c * blockDim.x + tx;

    if (ctx < size - sm_tour - 2) {
      // Movement indexes
      k = ctx >= by;
      i = k * (ctx - by + 1) + (!k) * (by - ctx);
      j = k * (ctx + 2) + (!k) * (size - ctx - sm_tour - 1);

      cost = (size - i) * (int(GPU_DIST_COORD(i - 1, j)) - sm_soldist[i]);
      // When computing PATHs and j = size-1, there's no j+1 element
      if (j + 1 < size)
        cost += (size - j - 1) *
                (int(GPU_DIST_COORD(i, j + 1)) - sm_soldist[j + 1]);

      for (k = 1; k <= j - i; k++)
        cost +=
            (size - i - k) * (int(sm_soldist[j - k + 1]) - sm_soldist[i + k]);

      k4printf("GPU_2OPT(%d,%d) = %d\n", i, j, cost);

      if (cost < sm_move[tx]) {
        sm_move[tx] = cost;
        sm_move[tx + csize] = GPU_MOVE_PACKID(i, j, MLMI_2OPT);
      }
    }
  }

  __syncthreads();

  /*
   * Minimum cost reduction
   */
  for (i = GPU_DIVCEIL(csize, 2); i > 1; i = GPU_DIVCEIL(i, 2)) {
    if (tx < i) {
      if ((tx + i < csize) && (sm_move[tx] > sm_move[tx + i])) {
        sm_move[tx] = sm_move[tx + i];
        sm_move[tx + csize] = sm_move[tx + csize + i];
      }
    }
    __syncthreads();
  }

  if (tx == 0) {
    // The first 2 elements was not compared
    if (sm_move[0] > sm_move[1]) {
      sm_move[0] = sm_move[1];
      sm_move[csize] = sm_move[csize + 1];
    }

    gm_move[by].w = GPU_MOVE_PACK64(sm_move[0], sm_move[csize]);
    k4printf("Block %d: GPU_2OPT(%u,%u) = %d\n", by, gm_move[by].s.i,
             gm_move[by].s.j, gm_move[by].s.cost);
  }

  /*
      // MIN cost reduction
      for(i=GPU_DIVCEIL(csize,2);i > 0;i >>= 1) {
          if(tx < i) {
              if(tx + i < csize) {
                  if(sm_move[tx] > sm_move[tx + i]) {
                      sm_move[tx] = sm_move[tx + i];
                      sm_move[csize + tx] = sm_move[csize + tx + i];
                  }
              }
          }
          __syncthreads();
      }

      if(tx == 0) {
          gm_move[by].w = GPU_MOVE_PACK64(sm_move[0],sm_move[csize]);
          k4printf("Block %d: MIN 2OPT(%u,%u) = %d\n",by,
                          gm_move[by].s.i,
                          gm_move[by].s.j,
                          gm_move[by].s.cost);
      }
  */
}
