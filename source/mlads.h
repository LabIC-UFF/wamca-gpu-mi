/**
 * @file   mlads.h
 *
 * @brief  Auxiliary Data Structure (ADS)
 *
 * @author Eyder Rios
 * @date   2014-06-01
 */

#ifndef __mlads_h
#define __mlads_h

#include "types.h"
#include "gpu.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

#define COST_INFTY          0x7fffffff

#define MLMI_OROPT(k)       int(MLMI_OROPT1 + k - 1)
#define MLMI_OROPT_K(id)    ( int(id) - 1 )
#define MLMI_KERNEL(m)      int((m) & 0xf)

// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

/*
 * ADS buffer
 *
 *     info      coords[x,y]        [W|Id]              T               C[0]                C[size-1]
 * +---------+----------------+----------------+----------------+----------------+-----+----------------+
 * | b elems | size elems |gap| size elems |gap| size elems |gap| size elems |gap| ... | size elems |gap|
 * +---------+----------------+----------------+----------------+----------------+-----+----------------+
 *
 * b = MLP_ADSINFO_ELEMS = GPU_GLOBAL_ALIGN/sizeof(uint) = 128/4 = 32 elems
 *
 * info[0]  : size (ADS buffer size in bytes)
 * info[1]  : rowElems
 * info[2]  : solElems (sol size)
 * info[3]  : solCost
 * info[4]  : tour
 * info[5]  : round
 * info[6]  : reserved
 * ...
 * info[b-1]: reserved
 */

#define ADS_INFO_SIZE           GPU_GLOBAL_ALIGN
#define ADS_INFO_ELEMS          ( ADS_INFO_SIZE / sizeof(uint) )

//#define ADS_INFO_ADS_SIZE       0
//#define ADS_INFO_ROW_ELEMS      1
//#define ADS_INFO_SOL_ELEMS      2
//#define ADS_INFO_SOL_COST       3
//#define ADS_INFO_TOUR           4
//#define ADS_INFO_ROUND          5

#define ADS_COORD_PTR(ads)      ( puint(ads + 1) )
#define ADS_SOLUTION_PTR(ads,r) ( puint(ads + 1) +   r )
#define ADS_TIME_PTR(ads,r)     ( puint(ads + 1) + 2*r )
#define ADS_COST_PTR(ads,r)     ( puint(ads + 1) + 3*r )


// ################################################################################# //
// ##                                                                             ## //
// ##                                 DATA TYPES                                  ## //
// ##                                                                             ## //
// ################################################################################# //

/*!
 * MLArch
 */
typedef enum  {
    MLSA_CPU,                   ///< CPU
    MLSA_SGPU,                  ///< Single GPU
    MLSA_MGPU,                  ///< Multi  GPU
} MLArch;

/*!
 * MLHeuristic
 */
typedef enum {
    MLSH_RVND,                  ///< RVND
    MLSH_DVND,                  ///< DVND
} MLHeuristic;

/*!
 * MLInprovement
 */
typedef enum  {
    MLSI_BEST,                  ///< Best improvement
    MLSI_MULTI,                 ///< Multi improvement
} MLImprovement;

/*!
 * MLHistorySelect
 */
typedef enum  {
    MLSS_BEST,                  ///< Best solution
    MLSS_RAND,                  ///< Random solution
} MLHistorySelect;

/*!
 * MLADSInfo
 */
struct MLADSInfo {
    uint     size;              ///< ADS buffer size in bytes
    uint     rowElems;          ///< ADS row  elems
    uint     solElems;          ///< Solution elems
    uint     solCost;           ///< Solution cost
    uint     tour;              ///< Tour (0/1)
    uint     round;             ///< Round: 0=0.0, 1=0.5
};

/*!
 * MLADSData
 */
union MLADSData {
    MLADSInfo   s;
    uint        v[ADS_INFO_ELEMS];
};

/*!
 * MLKernelId
 */
typedef enum {
    MLMI_SWAP,
    MLMI_2OPT,
    MLMI_OROPT1,
    MLMI_OROPT2,
    MLMI_OROPT3,
} MLMoveId;

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
    unsigned int    id : 4;
    unsigned int    i : 14;
    unsigned int    j : 14;
    int     cost;    // Goddamn Eyder...
};


/*!
 * MLMovePack
 */
union MLMovePack {
    MLMove64    s;
    ulong       w;
    int         i[2];
    uint        u[2];
    long        l[1];
};

/*!
 * MLMove
 */
struct MLMove {
    MLMoveId  id;
    int       i;
    int       j;
    int       cost;
};

/*!
 * MLPointer
 */
union MLPointer {
    const
    void        *p_cvoid;
    void        *p_void;
    char        *p_char;
    byte        *p_byte;
    short       *p_short;
    ushort      *p_ushort;
    int         *p_int;
    uint        *p_uint;
    long        *p_long;
    ulong       *p_ulong;
    llong       *p_llong;
    ullong      *p_ullong;
    MLMove      *p_move;
    MLMove64    *p_move64;
    MLMovePack  *p_mpack;
};

// ################################################################################# //
// ##                                                                             ## //
// ##                              GLOBAL VARIABLES                               ## //
// ##                                                                             ## //
// ################################################################################# //

/*!
 * Architecture names
 */
extern
const
char       *nameArch[];

extern
const
char       *nameLongArch[];

/*!
 * Heuristics names
 */
extern
const
char       *nameHeuristic[];

/*!
 * Movement names
 */
extern
const
char       *nameMove[];

/*!
 * Movement merge names
 */
extern
const
char       *nameImprov[];

extern
const
char       *nameLongImprov[];

/*!
 * Clear movement
 */
extern
const
MLMove64    MOVE64_NONE,
            MOVE64_INFTY;

extern
const
MLMove      MOVE_NONE,
            MOVE_INFTY;


// ################################################################################# //
// ##                                                                             ## //
// ##                                 FUNCTIONS                                   ## //
// ##                                                                             ## //
// ################################################################################# //

inline
void
move64ToMove(MLMove &m1, MLMove64 &m2) {
    m1.id = MLMoveId(m2.id);
    m1.i = m2.i;
    m1.j = m2.j;
    m1.cost = m2.cost;
}

inline
void
moveToMove64(MLMove64 &m1, MLMove &m2) {
    m1.id = m2.id;
    m1.i = m2.i;
    m1.j = m2.j;
    m1.cost = m2.cost;
}


#endif
