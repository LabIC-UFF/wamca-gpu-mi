/**
 * @file	vector.hpp
 *
 * @brief   Dynamic vector in both host and device memory.
 *
 * @author	Eyder Rios
 * @date	2011-11-30
 */


#ifndef __vector_h
#define __vector_h

#include <iostream>
#include <cuda/cuda.h>
#include "gpu.h"
#include "types.h"


namespace gpu {

template<typename T>
class Vector
{
public:
    int     flags;          ///< Vector flags
    uint    size;           ///< Number of elements
    T      *hData;          ///< Vector data on host
    T      *dData;          ///< Vector data on device

protected:
    /*!
     * Initialize class variables.
     */
    void
    init() {
        flags = DFLG_FREE_HOST | DFLG_FREE_DEVICE;
        size = 0;
        hData = NULL;
        dData = NULL;
    }

public:
    /*!
     * Create a vector of \a n elements.
     *
     * @param   n       Vector size
     */
    Vector(uint n = 0) {
        init();
        if(n > 0)
            resize(n);
    }
    /*!
     * Destroy vector releasing any allocated data.
     */
    ~Vector() {
        free();
    }
    /*!
     * Get host data pointer.
     *
     * @return  Returns pointer to data stored in host.
     */
    T *
    getHostData() {
        return hData;
    }
    /*!
     * Set host data pointer.
     *
     * @param   p       Pointer to data
     * @param   free    Auto free data pointed by \a p when instance is destroyed.
     */
    void
    setHostData(T *p, bool free = false) {
        hData = p;
        flags = free ? (flags | DFLG_FREE_HOST) : (flags & ~DFLG_FREE_HOST);
    }
    /*!
     * Get host data pointer.
     *
     * @return  Returns pointer to data stored in device.
     */
    T *
    getDeviceData() {
        return dData;
    }
    /*!
     * Set host data pointer.
     *
     * @param   p       Pointer to data
     * @param   free    Auto free data pointed by \a p when instance is destroyed.
     */
    void
    setDeviceData(T *p, bool free = false) {
        dData = p;
        flags = free ? (flags | DFLG_FREE_DEVICE) : (flags & ~DFLG_FREE_DEVICE);
    }
    /*!
     * Release any allocated data.
     */
    void
    free() {
        if(hData && (flags & DFLG_FREE_HOST))
            delete[] hData;
        if(dData && (flags & DFLG_FREE_DEVICE))
            cudaCheckErrors(cudaFree(dData));
        init();
    }
    /*!
      * Resize vector.
      * If new dimension \a n is equal to current vector dimension, nothing is done.
      *
      * @param   n       New vector dimension
      */
    void
    resize(uint n) {
        if(n == size)
            return;

        free();

        size = n;
        hData = new T[size];
        cudaCheckErrors(cudaMalloc(&dData,size*sizeof(T)));
    }
    /*!
     * Show vector in output stream \a os.
     * If \a os is not specified, then std::cout is used.
     *
     * @param   prompt  String to be shown before vector.
     * @param   os      Output stream
     */
    void
    show(const char *prompt = NULL, std::ostream &os = std::cout) {
        if(prompt)
            os << prompt;
        os << '<' << ' ';
        for(uint i=0;i < size;i++) {
            os << hData[i];
            if(i < size - 1)
                os << ',';
            else
                os << ' ';
        }
        os << '>' << '\n';
    }
    /*!
     * Copy data to host or device.
     */
    inline
    void
    copyTo(DataSource src) {
        if(src == DSRC_HOST)
            copyToHost();
        else
            copyToDevice();
    }
    /*!
     * Copy data from host to device.
     */
    inline
    void
    copyToDevice() {
        cudaCheckErrors(cudaMemcpy(dData,hData,size*sizeof(T),cudaMemcpyHostToDevice));
    }
    /*!
     * Copy data from host to device.
     */
    inline
    void
    copyToHost() {
        cudaCheckErrors(cudaMemcpy(hData,dData,size*sizeof(T),cudaMemcpyDeviceToHost));
    }
    /*!
     * Assign a same value to all elements of vector.
     *
     * @param   val     Value to be assigned.
     */
    inline
    void
    fill(T val) {
        for(uint i=0;i < size;i++)
            hData[i] = val;
    }
    /*!
     * Assign zero to all elements of vector.
     */
    inline
    void
    fillz() {
        memset(hData,0,size*sizeof(T));
    }
    /*!
     * Vector index operator (host only).
     *
     * @param   index   Element index
     * @return  Returns a reference for element in index \a index.
     */
    inline
    T &
    operator[](uint index) {
        return hData[index];
    }
};

/*
 * Vectors of basic types.
 */
typedef Vector<int>     VectorI;
typedef Vector<uint>    VectorU;
typedef Vector<long>    VectorL;
typedef Vector<ulong>   VectorUL;
typedef Vector<llong>   VectorLL;
typedef Vector<ullong>  VectorULL;

}

#endif	// __vector_h
