/**
 * @file    graph.hpp
 *
 * @brief   Implements a simple graph structure.
 *
 * @author  Eyder Rios
 * @date    2016-01-14
 */

#ifndef __graph_h
#define __graph_h

#include <iostream>
#include <string.h>
#include "types.h"
#include "except.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                             CONSTANTS & MACROS                             ## //
// ##                                                                            ## //
// ################################################################################ //

#define GRAPH_WORD_BYTES    sizeof(BitWord)
#define GRAPH_WORD_BITS     (8 * GRAPH_WORD_BYTES)


// ################################################################################ //
// ##                                                                            ## //
// ##                                 DATA TYPES                                 ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                CLASS Graph                                 ## //
// ##                                                                            ## //
// ################################################################################ //

template<typename T>
class Graph
{
private:
    typedef ullong  BitWord;

    union   Pointer {
        void    *p_void;
        byte    *p_byte;
        T       *p_vertex;
        BitWord *p_bitword;
    };

public:
    Pointer  buffer;
    int      bufferSize;

    T       *vertex;
    int      vertexMax;
    int      vertexCount;

    BitWord *edges;
    BitWord *flags;
    uint     bitWords;

private:
    void
    init() {
        buffer.p_void = NULL;
        bufferSize = 0;

        vertex = NULL;
        vertexMax = 0;
        vertexCount = 0;

        edges = NULL;
        flags = NULL;
        bitWords = 0;
    }

public:
    Graph(int size = 0) {
        init();
        if(size > 0)
            resize(size);
    }
    ~Graph() {
        free();
    }
    void
    resize(int size) {
    	printf("graph resize to %d\n", size);
        if(size != vertexMax)
            free();

        vertexMax = size;
        vertexCount = 0;

        /*
         * Compute how many words are necessary to store size bits.
         *
         * bitWords = ceil(size / GRAPH_WORD_BITS)
         */
        bitWords = (size + GRAPH_WORD_BITS - 1) / GRAPH_WORD_BITS;

        /*
         * buffer
         *
         *  +---------------+---------------+-----------------------+
         *  |  size elems   | rowSize elems |  size*rowWords elems  |
         *  +---------------+---------------+-----------------------+
         *  ^               ^               ^
         *  |               |               |
         *  vertex          flags           edges
         */
        // Buffer size in bytes
        bufferSize = size*sizeof(T) + bitWords*GRAPH_WORD_BYTES*(size + 1);
        buffer.p_byte = new byte[bufferSize];

        vertex = buffer.p_vertex;
        flags  = (BitWord *) (vertex + size);
        edges  = (BitWord *) (flags  + bitWords);

        clear();

#if 0
        printf("sizeof(T)\t= %lu\n",sizeof(T));
        printf("sizeof(BitWord)\t= %lu\n",sizeof(BitWord));
        printf("BitWord bits\t= %lu\n",8*sizeof(BitWord));
        printf("maxVertex\t= %u\n\n",size);

        printf("bufferSize\t= %u bytes\n",bufferSize);
        printf("bitWords\t= %u words = %lu bytes\n\n",bitWords,bitWords*GRAPH_WORD_BYTES);

        printf("buffer\t= %p\n",buffer.p_void);
        printf("vertex\t= %p\n",vertex);
        printf("flags\t= %p\n",flags);
        printf("edges\t= %p\n",edges);
#endif
    }
    void
    free() {
        if(buffer.p_byte)
            delete[] buffer.p_byte;

        init();
    }
    void
    clear() {
        vertexCount = 0;
        memset(buffer.p_byte,0,bufferSize);
    }
    void
    clearFlags() {
        memset(flags,0,bitWords*GRAPH_WORD_BYTES);
    }
    void
    clearEdges() {
        memset(flags,0,vertexMax*bitWords*GRAPH_WORD_BYTES);
    }
    void
    addVertex(const T &v, const bool flag = false) {
        ASSERT(vertexCount < vertexMax,
               "%s: Cannot add more vertex to graph",__FUNCTION__);

        vertex[vertexCount] = v;
        setFlag(vertexCount++,flag);
    }
    void
    getVertex(int i, T &v) {
        ASSERT(i < vertexCount,
               "%s: invalid index range (i=%d, count=%d)",__FUNCTION__,i,vertexCount);
        v = vertex[i];
    }
    bool
    getFlag(int i) {
        ASSERT(i < vertexCount,
               "%s: invalid index range (i=%d, count=%d)",__FUNCTION__,i,vertexCount);
        return flags[i / GRAPH_WORD_BITS] & (1ul << (i % GRAPH_WORD_BITS));
    }
    void
    setFlag(int i, short flag) {
        ASSERT(i < vertexCount,
               "%s: invalid index range (i=%d, count=%d)",__FUNCTION__,i,vertexCount);
        if(flag)
            flags[i / GRAPH_WORD_BITS] |=   1ul << (i % GRAPH_WORD_BITS);
        else
            flags[i / GRAPH_WORD_BITS] &= ~(1ul << (i % GRAPH_WORD_BITS));
    }
    void
    setEdge(int i, int j) {
        setArc(i,j);
        setArc(j,i);
    }
    void
    setArc(int i, int j) {
        ASSERT((i < vertexCount) && (j < vertexCount),
               "%s: invalid index range (i=%d, j=%d, count=%d)",__FUNCTION__,i,j,vertexCount);
        i = i*bitWords + (j / GRAPH_WORD_BITS);
        edges[i] |= 1ul << (j % GRAPH_WORD_BITS);
    }
    void
    delEdge(int i, int j) {
        delArc(i,j);
        delArc(j,i);
    }
    void
    delArc(int i, int j) {
        ASSERT((i < vertexCount) && (j < vertexCount),
               "%s: invalid index range (i=%d, j=%d, count=%d)",__FUNCTION__,i,j,vertexCount);
        i = i*bitWords + (j / GRAPH_WORD_BITS);
        edges[i] &= ~(1ul << (j % GRAPH_WORD_BITS));
    }
    int
    hasEdge(int i, int j) {
        ASSERT((i < vertexCount) && (j < vertexCount),
               "%s: invalid index range (i=%d, j=%d, count=%d)",__FUNCTION__,i,j,vertexCount);
        i = i*bitWords + (j / GRAPH_WORD_BITS);
        return (edges[i] & (1ul << (j % GRAPH_WORD_BITS))) > 0;
    }
    void
    showEdges(std::ostream &os = std::cout) {
        os << "EDGES:\n";
        for(int i=0;i < vertexCount;i++) {
            for(int j=0;j < vertexCount;j++)
                os << (hasEdge(i,j) ? 1 : 0) << ' ';
            os << std::endl;
        }
        os << std::endl;
    }
    void
    showFlags(std::ostream &os = std::cout) {
        os << "FLAGS:\n";
        for(int i=0;i < vertexCount;i++)
            os << (getFlag(i) ? 1 : 0) << ' ';
        os << std::endl;
    }
    void
    writeDot(std::ostream &os = std::cout) {
        os << "graph G {\n\tnode [shape=circle,fontsize=10];\n";
        for(int i=0;i < vertexCount;i++) {
            for(int j=i + 1;j < vertexCount;j++) {
                if(hasEdge(i,j))
                    os << '\t' << i << " -- " << j << ";\n";
            }
        }
        os << "}\n";
    }
    T &
    operator[](int i) {
        ASSERT(i < vertexCount,
               "%s: invalid index range (i=%d, count=%d)",__FUNCTION__,i,vertexCount);
        return vertex[i];
    }
};

#endif  // __graph_hpp
