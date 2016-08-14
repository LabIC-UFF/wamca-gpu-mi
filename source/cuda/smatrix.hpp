/**
 * @file   smatrix.h
 *
 * @brief  Implements an dynamic square matrix in both host and device.
 *
 * @author Eyder Rios
 * @date   2014-11-24
 */

#ifndef __smatrix_h
#define __smatrix_h

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <cuda/cuda.h>
#include "log.hpp"
#include "types.h"
#include "gpu.h"


namespace gpu {

/*!
 * Host Square Matrix
 */
template<typename T>
class SquareMatrix
{
public:
    uint      size;             ///< number of elements
    uint      dim;              ///< matrix dimension
    uint      width;            ///< width of each printed element
    T        *hData;            ///< matrix elements (host)
    T       **hRows;            ///< address of first element on each line (host)
    T        *dData;            ///< matrix elements (device)
    T       **dRows;            ///< address of first element on each line (device)

protected:
    /*!
     * Initialize class variables.
     */
    void
    init() {
        size = 0;
        dim  = 0;
        width = 5;
        hData = NULL;
        hRows = NULL;
        dData = NULL;
        dRows = NULL;
    }

public:
    /*!
     * Create a square matrix of dimension \a n.
     *
     * @param   n       Matrix dimension
     */
    SquareMatrix(uint n = 0) {
        init();
        if(n > 0)
            redim(n);
    }
    /*!
     * Create an square matrix as copy of another (copy constructor).
     */
    SquareMatrix(const SquareMatrix<T> &m) {
        init();
        assign(m);
    }
    /*!
     * Destroy matrix releasing any allocated data.
     */
    ~SquareMatrix() {
        free();
    }
    /*!
     * Set element width.
     */
    void
    setWidth(uint w) {
        width = w;
    }
    /*!
     * Get host data pointer.
     *
     * @return  Returns pointer to data stored in host.
     */
    T *
    hostData() {
        return hRows;
    }
    /*!
     * Get host data pointer.
     *
     * @return  Returns pointer to data stored in device.
     */
    T *
    deviceData() {
        return dRows;
    }
    /*!
     * Resize matrix dimension.
     * If new dimension \a n is equal to current matrix dimension, nothing is done.
     *
     * @param   n       New matrix dimension
     */
    void
    redim(uint n) {
        if(dim == n)
          return;

        free();

        dim  = n;
        size = n * n;

        hData = new T[size];
        hRows = new T *[size];

        cudaCheckErrors(cudaMalloc(&dData,size*sizeof(T)));
        cudaCheckErrors(cudaMalloc(&dRows,size*sizeof(T *)));

        for(uint i=0;i < dim;i++)
            hRows[i] = dData + dim*i;

        cudaCheckErrors(cudaMemcpy(dRows,hRows,size*sizeof(T *),cudaMemcpyHostToDevice));

        for(uint i=0;i < dim;i++)
            hRows[i] = hData + dim*i;
    }
    /*!
     * Release matrix allocated memory.
     * After this call the matrix will be empty.
     */
    void
    free() {
        if(dRows)
            checkCudaErrors(cudaFree(dRows));
        if(dData)
            checkCudaErrors(cudaFree(dData));

        if(hRows)
            delete[] hRows;
        if(hData)
            delete[] hData;


        init();
    }
    /*!
     * Show matrix in output stream \a os.
     * If \a os is not specified, then std::cout is used.
     *
     * @param   os      Output stream
     */
    void
    show(std::ostream &os = std::cout) {
        for(uint i=0;i < dim;i++) {
            os << '|';
            for(uint j=0;j < dim;j++)
                os << std::setw(width) << hRows[i][j];
            os << '|' << '\n';
        }
        os << '\n';
    }
    /*!
     * Assign an square matrix.
     *
     * @param   m   Source square matrix.
     * @return  Returns a reference to this matrix.
     */
    SquareMatrix<T> &
    assign(SquareMatrix<T> &m) {
        assert(m.dim == dim);
        cudaCheckErrors(cudaMemcpy(hData,m.hData,size*sizeof(T),cudaMemcpyHostToHost));
        cudaCheckErrors(cudaMemcpy(dData,m.dData,size*sizeof(T),cudaMemcpyDeviceToDevice));
        return *this;
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
     * Assign a same value to all elements of matrix.
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
     * Assign zero to all elements of matrix.
     */
    inline
    void
    fillz() {
        memset(hData,0,size*sizeof(T));
    }
    /*!
     * Matrix index operator (host only).
     *
     * @param   row     Matrix row
     * @return  Returns a reference for first element of \a row line.
     */
    inline
    T *&
    operator[](uint row) {
        return hRows[row];
    }
};

/*
 * Matrixes of basic types.
 */
typedef SquareMatrix<int>     SMatrixI;
typedef SquareMatrix<uint>    SMatrixU;
typedef SquareMatrix<long>    SMatrixL;
typedef SquareMatrix<ulong>   SMatrixUL;
typedef SquareMatrix<llong>   SMatrixLL;
typedef SquareMatrix<ullong>  SMatrixULL;

}

#endif // __matrix_h
