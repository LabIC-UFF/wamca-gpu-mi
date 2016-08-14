/*!
 * @file   tsplib.cpp
 *
 * @brief  TSPLIB library source file
 *
 * @author Eyder Rios
 * @date   2011-09-12
 */

#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "except.h"
#include "tsplib.h"

namespace tsplib {

using namespace std;


/*
 *******************************************************************************************************************
 *
 * TSPLIB local macros
 *
 *******************************************************************************************************************
 */

/*
 * Expand macro as string. Example:
 *   EXPNULL(PFX,NAME) is expanded as: { PFX_NAME,  NULL   }
 *   EXPSTR(PFX,NAME)  is expanded as: { PFX_NAME, "NAME" }
 */
#define EXPSTRS(pfx,id)     { pfx##_##id, #id "_SECTION"  }
#define EXPSTR(pfx,id)      { pfx##_##id, #id  }
#define EXPNUL(pfx,id)      { pfx##_##id, NULL }

/*
 *******************************************************************************************************************
 *
 * TSPLIB local variables
 *
 *******************************************************************************************************************
 */

/*!
 * TSPLIB data file fields types information
 */
FieldInfo fieldTypeInfo[] = {
    EXPNUL(TFT,NONE),                   // Expanded as: { TFT_NONE,   NULL  }
    EXPSTR(TFT,NAME),                   // Expanded as: { TFT_NAME,  "NAME" }
    EXPSTR(TFT,TYPE),
    EXPSTR(TFT,COMMENT),
    EXPSTR(TFT,DIMENSION),
    EXPSTR(TFT,CAPACITY),
    EXPSTR(TFT,EDGE_WEIGHT_TYPE),
    EXPSTR(TFT,EDGE_WEIGHT_FORMAT),
    EXPSTR(TFT,EDGE_DATA_FORMAT),
    EXPSTR(TFT,NODE_COORD_TYPE),
    EXPSTR(TFT,DISPLAY_DATA_TYPE),
    EXPNUL(TFT,NONE),
};

/*!
 * TSPLIB data file problem types information
 */
FieldInfo problemTypeInfo[] = {
    EXPNUL(TPT,NONE),
    EXPSTR(TPT,TSP),
    EXPSTR(TPT,ATSP),
    EXPSTR(TPT,SOP),
    EXPSTR(TPT,HCP),
    EXPSTR(TPT,CVRP),
    EXPSTR(TPT,TOUR),
    EXPNUL(TPT,NONE),
};

/*!
 * TSPLIB data file edge weight types information
 */
FieldInfo edgeWeightTypeInfo[] = {
    EXPNUL(TWT,NONE),
    EXPSTR(TWT,EXPLICIT),
    EXPSTR(TWT,EUC_2D),
    EXPSTR(TWT,EUC_3D),
    EXPSTR(TWT,MAX_2D),
    EXPSTR(TWT,MAX_3D),
    EXPSTR(TWT,MAN_2D),
    EXPSTR(TWT,MAN_3D),
    EXPSTR(TWT,CEIL_2D),
    EXPSTR(TWT,GEO),
    EXPSTR(TWT,ATT),
    EXPSTR(TWT,XRAY1),
    EXPSTR(TWT,XRAY2),
    EXPSTR(TWT,SPECIAL),
    EXPNUL(TWT,NONE),
};

/*!
 * TSPLIB data file edge weight formats information
 */
FieldInfo edgeWeightFormatInfo[] = {
    EXPNUL(TWF,NONE),
    EXPSTR(TWF,FUNCTION),
    EXPSTR(TWF,FULL_MATRIX),
    EXPSTR(TWF,UPPER_ROW),
    EXPSTR(TWF,LOWER_ROW),
    EXPSTR(TWF,UPPER_DIAG_ROW),
    EXPSTR(TWF,LOWER_DIAG_ROW),
    EXPSTR(TWF,UPPER_COL),
    EXPSTR(TWF,LOWER_COL),
    EXPSTR(TWF,UPPER_DIAG_COL),
    EXPSTR(TWF,LOWER_DIAG_COL),
    EXPNUL(TWF,NONE),
};

/*!
 * TSPLIB data file edge data formats information
 */
FieldInfo edgeDataFormatInfo[] = {
    EXPNUL(TDF,NONE),
    EXPSTR(TDF,EDGE_LIST),
    EXPSTR(TDF,ADJ_LIST),
    EXPNUL(TDF,NONE),
};

/*!
 * TSPLIB data file node coordinates types information
 */
FieldInfo nodeCoordTypeInfo[] = {
    EXPNUL(TNC,NONE),
    EXPSTR(TNC,TWOD_COORDS),
    EXPSTR(TNC,THREED_COORDS),
    EXPNUL(TNC,NONE),
};

/*!
 * TSPLIB data file display data types information
 */
FieldInfo displayDataTypeInfo[] = {
    EXPNUL(TDT,NONE),
    EXPSTR(TDT,COORD_DISPLAY),
    EXPSTR(TDT,NO_DISPLAY),
    EXPNUL(TDT,NONE),
};

/*!
 * TSPLIB data file data section types information
 */
FieldInfo dataSectionInfo[] = {
    EXPNUL(TDS,NONE),
    EXPSTRS(TDS,NODE_COORD),
    EXPSTRS(TDS,DEPOT),
    EXPSTRS(TDS,DEMAND),
    EXPSTRS(TDS,EDGE_DATA),
    EXPSTRS(TDS,FIXED_EDGE),
    EXPSTRS(TDS,DISPLAY_DATA),
    EXPSTRS(TDS,TOUR),
    EXPSTRS(TDS,EDGE_WEIGHT),
    EXPNUL(TDS,NONE),
};

/*
 *******************************************************************************************************************
 *
 * TSPLIB functions
 *
 *******************************************************************************************************************
 */

int getFieldInfoType(const char *fname, FieldInfo *finfo)
{
    int  i;

    for(i=1;finfo[i].fi_name;i++) {
        if(strcmp(finfo[i].fi_name,fname) == 0)
        return i;
    }
    return 0;
}

/*
 *******************************************************************************************************************
 *
 * TSPLibInstance
 *
 *******************************************************************************************************************
 */

TSPLibInstance::TSPLibInstance(bool flag)
{
    type = TPT_NONE;
    dimension = 0;
    capacity = 0;
    ewtype = TWT_NONE;
    ewformat = TWF_NONE;
    edformat = TDF_NONE;
    nctype = TNC_NONE;
    ddtype = TDT_NONE;

    freeMem  = flag;
    nodeData = NULL;
    distCalc = NULL;
}

TSPLibInstance::~TSPLibInstance()
{
    if(freeMem)
        free();
}

void
TSPLibInstance::alloc()
{
    free();

    nodeData = new NodeData[dimension];
    for(uint i=0;i < dimension;i++)
        nodeData[i].weights = new ushort[dimension];
}

void
TSPLibInstance::free()
{
    if(nodeData) {
        for(uint i=0;i < dimension;i++)
            delete[] nodeData[i].weights;
        delete[] nodeData;

        nodeData = NULL;
    }
}

int
TSPLibInstance::inear(Float n)
{
    return (n >= 0) ? (int) (n + TSP_CONST(0.5)) : (int) (n - TSP_CONST(0.5));
}

int
TSPLibInstance::ifloor(Float x)
{
    return (int) x;
}

uint
euclidianDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    return toInt(TSP_SQRT(TSP_SQR(node1->x - node2->x) + TSP_SQR(node1->y - node2->y)));
}

uint
pseudoEuclidianDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    Float  d;
    uint   t;

    d  = TSP_SQR(node1->x - node2->x) + TSP_SQR(node1->y - node2->y);
    d  = TSP_SQRT(d / TSP_CONST(10.0));

    t  = toInt(d);
    if(t < d)
        t++;

    return t;
}

uint ceilEuclidianDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    return (uint) ceil(TSP_SQRT(TSP_SQR(node1->x - node2->x) + TSP_SQR(node1->y - node2->y)));
}

uint
manhattanDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    Float   d = 0.0;
    int     i;

    d = abs(node1->x - node2->x) + abs(node1->y - node2->y);
    d = TSPLibInstance::inear(d);

    return toInt(d);
}

uint
maximumDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    uint  d,max;

    d   = toInt(::abs(node1->x - node2->x));
    max = d;
    d   = toInt(::abs(node1->y - node2->y));
    if(d > max)
        max = d;

    return max;
}

/*
 http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/TSPFAQ.html

 Q: I get wrong distances for problems of type GEO.
 A: There has been some confusion of how to compute the distances. I use the following code.

 For converting coordinate input to longitude and latitude in radian:
 PI = 3.141592;

 deg = (int) x[i];
 min = x[i]- deg;
 rad = PI * (deg + 5.0 * min / 3.0) / 180.0;

 For computing the geographical distance:

 RRR = 6378.388;

 q1 = cos( longitude[i] - longitude[j] );
 q2 = cos( latitude[i] - latitude[j] );
 q3 = cos( latitude[i] + latitude[j] );
 dij = (int) ( RRR * acos( 0.5*((1.0+q1)*q2 - (1.0-q1)*q3) ) + 1.0);

 */
uint
geoDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    Float   PI  =    TSP_CONST(3.141592),
            RRR = TSP_CONST(6378.388000);
    int     deg;
    Float   min,
            lat1,lat2,
            lon1,lon2,
            q1,q2,q3;

    // node1
    deg = (int) node1->x;
    min = node1->x - deg;
    lat1 = PI * (deg + TSP_CONST(5.0)*min / TSP_CONST(3.0)) / TSP_CONST(180.0);

    deg = (int) node1->y;
    min = node1->y - deg;
    lon1 = PI * (deg + TSP_CONST(5.0)*min / TSP_CONST(3.0)) / TSP_CONST(180.0);

    // node2
    deg = (int) node2->x;
    min = node2->x - deg;
    lat2 = PI * (deg + TSP_CONST(5.0)*min / TSP_CONST(3.0)) / TSP_CONST(180.0);

    deg = (int) node2->y;
    min = node2->y - deg;
    lon2 = PI * (deg + TSP_CONST(5.0)*min / TSP_CONST(3.0)) / TSP_CONST(180.0);

    // distance in Km
    q1 = TSP_COS(lon1 - lon2);
    q2 = TSP_COS(lat1 - lat2);
    q3 = TSP_COS(lat1 + lat2);

    deg = (RRR * TSP_ACOS(TSP_CONST(0.5)*((TSP_CONST(1.0) + q1)*q2 - (TSP_CONST(1.0) - q1)*q3)) + TSP_CONST(1.0));

    return toInt(deg);
}

uint
cristallographyDistance(NodeData *node1, NodeData *node2, FloatRound toInt)
{
    return TSP_CONST(0.0);
}

DistanceFunc
TSPLibInstance::getDistanceFunction()
{
    DistanceFunc func;

    func = NULL;
    switch(ewtype) {
    case TWT_EXPLICIT:
        break;
    case TWT_EUC_2D:
    case TWT_EUC_3D:
        func = euclidianDistance;
        break;
    case TWT_MAX_2D:
    case TWT_MAX_3D:
        func = maximumDistance;
        break;
    case TWT_MAN_2D:
    case TWT_MAN_3D:
        func = manhattanDistance;
        break;
    case TWT_CEIL_2D:
        func = ceilEuclidianDistance;
        break;
    case TWT_GEO:
        func = geoDistance;
        break;
    case TWT_ATT:
        func = pseudoEuclidianDistance;
        break;
    case TWT_XRAY1:
    case TWT_XRAY2:
        func = cristallographyDistance;
        break;
    case TWT_NONE:
    case TWT_SPECIAL:
        break;
    }

    return func;
}

bool
TSPLibInstance::isValid(ifstream &fin)
{
    FieldType    ft;
    DataSection  ds;
    char         buffer[1024],
                *token;
    int          i,tags;


    tags = 0;
    for(i=0;i < 4;i++) {
        if(fin.getline(buffer,sizeof(buffer))) {
            if((token = strtok(buffer,":")) != NULL) {
                ft = (FieldType) getFieldInfoType(token,fieldTypeInfo);
                if(ft == TFT_NONE) {
                    ds = (DataSection) getFieldInfoType(token,dataSectionInfo);
                    if(ds == TDS_NONE) {
                        tags = 0;
                        break;
                    }
                    else
                        tags++;
                }
                else
                    tags++;
            }
        }
    }

    return tags > 0;
}

bool
TSPLibInstance::isValid(const char *fname)
{
    ifstream    fin;

    fin.open(fname);
    if(!fin.is_open())
        return false;

    return isValid(fin);
}

//#pragma GCC diagnostic ignored "-Wunused-result"

bool
TSPLibInstance::loadNodeCoord(ifstream &fin)
{
    NodeData  *c;
    uint       i,j,d;

    alloc();

    c = nodeData;
    for(i=0;i < dimension;i++) {
        fin >> c->id;
        fin >> c->x;
        fin >> c->y;

        c->id--;

        c++;
    }

    for(i=0;i < dimension;i++) {
        for(j=i;j < dimension;j++) {
            if(i != j) {

#define TSPLIB_FLOOR
#ifdef  TSPLIB_FLOOR
                d = distCalc(nodeData + i,nodeData + j,TSPLibInstance::ifloor);
#else
                d = distCalc(nodeData + i,nodeData + j,TSPLibInstance::inear);
#endif

            }
            else
                d = 0;

            nodeData[i].weights[j] = nodeData[j].weights[i] = d;
        }
    }

    return true;
}

bool
TSPLibInstance::loadEdgeWeight(ifstream &fin)
{
    uint    i,j,d;

    alloc();

    d = 0;
    switch(ewformat) {
    case TWF_FULL_MATRIX:
        for(i=0;i < dimension;i++) {
            nodeData[i].id = i;
            for(j=0;j < dimension;j++)
                fin >> nodeData[i].weights[j];
        }
        break;

    // Rows of upper triangular matrix (w/  diagonal)
    case TWF_UPPER_ROW:
        d = 1;
        /* no break */

    // Rows of upper triangular matrix (w/o diagonal)
    case TWF_UPPER_DIAG_ROW:
        for(i=0;i < dimension;i++) {
            nodeData[i].id = i;
            for(j=i+d;j < dimension;j++) {
                fin >> nodeData[i].weights[j];
                nodeData[j].weights[i] = nodeData[i].weights[j];
            }
        }
        break;
        // Rows of lower triangular matrix (w/  diagonal)
    case TWF_LOWER_DIAG_ROW:
        d = 1;
        /* no break */

        // Rows of lower triangular matrix (w/o diagonal)
    case TWF_LOWER_ROW:
        for(i=0;i < dimension;i++) {
            nodeData[i].id = i;
            for(j=0;j < i+d;j++) {
                fin >> nodeData[i].weights[j];
                nodeData[j].weights[i] = nodeData[i].weights[j];
            }
        }
        break;
    case TWF_UPPER_DIAG_COL:
        d = 1;
        /* no break */

    // Columns of upper triangular matrix
    case TWF_UPPER_COL:
        for(i=0;i < dimension;i++) {
            nodeData[i].id = i;
            for(j=0;j < i+d;j++) {
                fin >> nodeData[i].weights[j];
                nodeData[j].weights[i] = nodeData[i].weights[j];
            }
        }
        break;

   // Columns of lower triangular matrix
   case TWF_LOWER_COL:
        d = 1;
        /* no break.* */

    case TWF_LOWER_DIAG_COL:
        for(i=0;i < dimension;i++) {
            nodeData[i].id = i;
            for(j=i+d;j < dimension;j++) {
                fin >> nodeData[i].weights[j];
                nodeData[j].weights[i] = nodeData[i].weights[j];
            }
        }
        break;
    default:
        break;
    }

    return true;
}

//#pragma GCC diagnostic warning "-Wunused-result"

char *
trim(char *s)
{
    char *p;
    int   n;

    p = s;
    while(*s && (*s == ' '))
        s++;

    if((n = strlen(s)) > 0) {
        p = s + n - 1;
        while(*p && (*p == ' '))
            p--;
        *(p + 1) = '\0';
    }

    return s;
}

bool
TSPLibInstance::load(const char *fname, FloatRound round)
{
    ifstream fin;
    char     buffer[1024],
            *token,
            *data;
    bool     flag;
    int      n;

    fin.open(fname);
    if(!fin.is_open())
        return false;

    distCalc = euclidianDistance;

    flag = true;
    while(flag && fin.getline(buffer,sizeof(buffer))) {

        if((token = strtok(buffer,": \n")) == NULL)
            break;

        if((data = strtok(NULL," \n")) != NULL)
            data = trim(data);

        switch(getFieldInfoType(token,fieldTypeInfo)) {
        case TFT_NAME:
            name = data;
            break;
        case TFT_TYPE:
            type = (ProblemType) getFieldInfoType(data,problemTypeInfo);
            break;
        case TFT_COMMENT:
            comment = data;
            break;
        case TFT_DIMENSION:
            stringstream(data) >> dimension;
            break;
        case TFT_CAPACITY:
            stringstream(data) >> capacity;
            break;
        case TFT_EDGE_WEIGHT_TYPE:
            ewtype = (EdgeWeightType) getFieldInfoType(data,edgeWeightTypeInfo);
            distCalc = getDistanceFunction();
            break;
        case TFT_EDGE_WEIGHT_FORMAT:
            ewformat = (EdgeWeightFormat) getFieldInfoType(data,edgeWeightFormatInfo);
            break;
        case TFT_EDGE_DATA_FORMAT:
            edformat = (EdgeDataFormat) getFieldInfoType(data,edgeDataFormatInfo);
            break;
        case TFT_NODE_COORD_TYPE:
            nctype = (NodeCoordType) getFieldInfoType(data,nodeCoordTypeInfo);
            break;
        case TFT_DISPLAY_DATA_TYPE:
            ddtype = (DisplayDataType) getFieldInfoType(data,displayDataTypeInfo);
            break;
        default:
            switch(getFieldInfoType(token,dataSectionInfo)) {
            case TDS_NODE_COORD:
                loadNodeCoord(fin);
                break;
            case TDS_EDGE_WEIGHT:
                loadEdgeWeight(fin);
                break;
            case TDS_DEPOT:
            case TDS_DEMAND:
            case TDS_EDGE_DATA:
            case TDS_FIXED_EDGE:
            case TDS_DISPLAY_DATA:
            case TDS_TOUR:
                EXCEPTION("Not implemented (data section)");
                break;
            case TDS_NONE:
                flag = false;
                break;
            }
            flag = false;
            break;
        }
    }

    return true;
}

#if 0

bool tsplibTest(char *baseDir)
{
    TSPLibTestbed test[] = {
            { "pcb442.tsp",  221440 },
            { "gr666.tsp",   423710 },
            { "att532.tsp",  309636 },
            { NULL,               0 },
    };
    TSPLibInstance tsplib;
    Path           fname;
    uint           k,i,j,w,cost;

    for(k=0;test[k].fname;k++) {
        tsplibInit(&tsplib);

        pathCopy(fname,baseDir);
        pathConcat(fname,test[k].fname);

        printf("Loading instance %s: %s\n",test[k].fname,fname);
        if(tsplibLoad(&tsplib,fname,false)) {

            printf("Calculating TSP cost of solution (0,1,2,...,%u,0):\n",tsplib.tsp->nodeCount-1);

            cost = 0;
            for(i=0;i < tsplib.tsp->nodeCount;i++) {
                j = (i < tsplib.tsp->nodeCount - 1) ? i + 1 : 0;

                w = tsplib.tsp->nodes[i].weight[j];
                cost += w;

                //printf("\t(%u,%u) = %u\n",tsplib.tsp->nodes[i].id,tsplib.tsp->nodes[j].id,w);
            }

            printf("\tCalculated cost\t: %u\n",cost);
            printf("\tCorrect    cost\t: %u\n",test[k].cost);
            printf("\tResult\t\t: %s\n\n",(cost == test[k].cost) ? "OK" : "ERROR");
        }

        tsplibFree(&tsplib);
    }


    return true;
}

#endif

}
