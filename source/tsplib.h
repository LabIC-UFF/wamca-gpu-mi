/*!
 * @file   tsplib.h
 *
 * @brief  TSPLIB library header file
 *
 * @author Eyder Rios
 * @date   2011-09-12
 */

#ifndef __tsplib_h
#define __tsplib_h

#include <fstream>
#include <string>
#include "types.h"

namespace tsplib {

#define TLIB_COMMENT_LEN            127

/*!
 * Store TSPLIB data file fields information.
 */
typedef struct {
    int         fi_type;    //<! Field type id
    const char *fi_name;    //<! Field identifier
} FieldInfo;

/*!
 * TSPLIB float datatype
 */
#define TSP_PREC_FLOAT

#ifdef  TSP_PREC_FLOAT

#define TSP_CONST(c)        (c##F)
#define TSP_SQR(x)          (x)*(x)
#define TSP_SQRT(x)         sqrtf(x)
#define TSP_CEIL(x)         ceilf(x)
#define TSP_COS(x)          cosf(x)
#define TSP_ACOS(x)         acosf(x)

#else

typedef double  Float;

#define TSP_CONST(c)        (c)
#define TSP_SQR(x)          (x)*(x)
#define TSP_SQRT(x)         sqrt(x)
#define TSP_CEIL(x)         ceil(x)
#define TSP_COS(x)          cos(x)
#define TSP_ACOS(x)         acos(x)

#endif

typedef float   Float;

/*!
 * TSPLIB node coordinates
 */
typedef struct {
    ushort   id;
    Float    x;
    Float    y;
    ushort  *weights;
} NodeData;

/*!
 * TSPLIB float round function
 */
typedef int (*FloatRound)(Float);

/*!
 * TSPLIB distance function pointer.
 */
typedef uint (*DistanceFunc)(NodeData *node1, NodeData *node2, FloatRound toInt);

/*!
 * TSPLIB data file fields types information
 */

typedef enum {
    TFT_NONE,
    TFT_NAME,
    TFT_TYPE,
    TFT_COMMENT,
    TFT_DIMENSION,
    TFT_CAPACITY,
    TFT_EDGE_WEIGHT_TYPE,
    TFT_EDGE_WEIGHT_FORMAT,
    TFT_EDGE_DATA_FORMAT,
    TFT_NODE_COORD_TYPE,
    TFT_DISPLAY_DATA_TYPE,
} FieldType;


/*!
 * TSPLIB data file weight types information
 */

typedef enum {
    TWT_EUCLIDIAN,
    TWT_PEUCLIDIAN,
    TWT_CEUCLIDIAN,
    TWT_MANHATTAN,
    TWT_MAXIMUM,
    TWT_GEOGRAPHIC,
    TWT_CRYSTAL,
} WeightType;

/*!
 * TSPLIB data file problem types information
 */

typedef enum {
    TPT_NONE,
    TPT_TSP,
    TPT_ATSP,
    TPT_SOP,
    TPT_HCP,
    TPT_CVRP,
    TPT_TOUR,
} ProblemType;

/*!
 * TSPLIB data file edge weight types information
 */

typedef enum {
    TWT_NONE,
    TWT_EXPLICIT,
    TWT_EUC_2D,
    TWT_EUC_3D,
    TWT_MAX_2D,
    TWT_MAX_3D,
    TWT_MAN_2D,
    TWT_MAN_3D,
    TWT_CEIL_2D,
    TWT_GEO,
    TWT_ATT,
    TWT_XRAY1,
    TWT_XRAY2,
    TWT_SPECIAL,
} EdgeWeightType;

/*!
 * TSPLIB data file edge weight formats information
 */

typedef enum {
    TWF_NONE,
    TWF_FUNCTION,
    TWF_FULL_MATRIX,
    TWF_UPPER_ROW,
    TWF_LOWER_ROW,
    TWF_UPPER_DIAG_ROW,
    TWF_LOWER_DIAG_ROW,
    TWF_UPPER_COL,
    TWF_LOWER_COL,
    TWF_UPPER_DIAG_COL,
    TWF_LOWER_DIAG_COL,
} EdgeWeightFormat;

/*!
 * TSPLIB data file edge data formats information
 */

typedef enum {
    TDF_NONE,
    TDF_EDGE_LIST,
    TDF_ADJ_LIST,
} EdgeDataFormat;

/*!
 * TSPLIB data file node coordinates types information
 */

typedef enum {
    TNC_NONE,
    TNC_TWOD_COORDS,
    TNC_THREED_COORDS,
} NodeCoordType;

typedef enum {
    TDT_NONE,
    TDT_COORD_DISPLAY,
    TDT_TWOD_DISPLAY,
    TDT_NO_DISPLAY,
} DisplayDataType;

/*!
 * TSPLIB data file data section types information
 */

typedef enum {
    TDS_NONE,
    TDS_NODE_COORD,
    TDS_DEPOT,
    TDS_DEMAND,
    TDS_EDGE_DATA,
    TDS_FIXED_EDGE,
    TDS_DISPLAY_DATA,
    TDS_TOUR,
    TDS_EDGE_WEIGHT,
} DataSection;

/*!
 * Handle a TSPLIB instance
 */
class TSPLibInstance
{
public:
    ProblemType       type;             ///< instance type
    std::string       name;             ///< instance name
    std::string       comment;          ///< comment
    EdgeWeightType    ewtype;           ///< edge weight type
    EdgeWeightFormat  ewformat;         ///< edge weight format
    EdgeDataFormat    edformat;         ///< edge data format
    NodeCoordType     nctype;           ///< node coordinate type
    DisplayDataType   ddtype;           ///< display data type
    uint              capacity;         ///< capacity of vehicles
    uint              dimension;        ///< problem dimension
    DistanceFunc      distCalc;         ///< function for distance calculation
    NodeData         *nodeData;         ///< nodes data
    bool              freeMem;          ///< Release memory when object is destroyed

public:
    static
    int
    inear(Float n);

    static
    int
    ifloor(Float x);

protected:
    /*!
     * Load clients coordinates.
     */
    bool
    loadNodeCoord(std::ifstream &fin);
    /*!
     *
     */
    bool
    loadEdgeWeight(std::ifstream &fin);
    /*!
     * Alloc memory for nodes data.
     */
    void
    alloc();
    /*!
     * Release memory allocated for nodes data
     */
    void
    free();

public:
    /*!
     * Initialize a TSPLibInstance instance.
     *
     * @param  tsplib        Pointer to instance
     */
    TSPLibInstance(bool free = false);
   ~TSPLibInstance();
    /*!
     * Create an instance of TSPRandInstance loading data from a file if given.
     *
     *  The \a round parameter points to a function to convert float values to integers.
     *  If \a round is \a NULL, then float values are truncated: (int) value
     *
     * @param   fname       instance file name
     *
     * @see     FloatRound
     */
    bool
    load(const char *fname, FloatRound round = NULL);
    /*!
     * Get a pointer to distance calcultion function.
     */
    DistanceFunc
    getDistanceFunction();
    /*!
     * Tests whether a file is in TSPLIB format.
     *
     * @param  baseDir      directory containing TSPLIB files
     *
     * @return Returns \a true whether file is valid, otherwise \a false.
     */
    bool
    isValid(const char *fname);
    /*!
     * Tests whether a file is in TSPLIB format.
     *
     * @param  fin          Instance file stream
     *
     * @return Returns \a true whether file is valid, otherwise \a false.
     */
    bool
    isValid(std::ifstream &fin);
};

}

#endif // __tsplib_h
