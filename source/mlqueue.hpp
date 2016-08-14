/**
 * @file	mlqueue.hpp
 *
 * @brief	Implements a circular queue.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef __mlqueue_hpp
#define __mlqueue_hpp

#include <pthread.h>
#include <iostream>
#include "except.h"
#include "utils.h"


// ################################################################################ //
// ##                                                                            ## //
// ##                               CONSTANTS & MACROS                           ## //
// ##                                                                            ## //
// ################################################################################ //

// ################################################################################ //
// ##                                                                            ## //
// ##                                  DATA TYPES                                ## //
// ##                                                                            ## //
// ################################################################################ //


// ################################################################################ //
// ##                                                                            ## //
// ##                                CLASS MLQueue                               ## //
// ##                                                                            ## //
// ################################################################################ //

template<typename T, int size>
class MLLockQueue
{
private:
    uint      counter;
    uint      first;
    uint      last;
    T         buffer[size];

    pthread_mutex_t mtxQueue;
    pthread_cond_t  cvxQueue;

public:
    MLLockQueue() {
        pthread_mutex_init(&mtxQueue,NULL);
        pthread_cond_init(&cvxQueue,NULL);
        clear();
    }
    ~MLLockQueue() {
        pthread_mutex_destroy(&mtxQueue);
        pthread_cond_destroy(&cvxQueue);
    }
    inline
    void
    clear() {
        pthread_mutex_lock(&mtxQueue);
        counter = 0;
        first = 0;
        last  = 0;
        pthread_mutex_unlock(&mtxQueue);
    }
    void
    push(const T &data) {
        pthread_mutex_lock(&mtxQueue);

        if(counter == size)
            EXCEPTION("Cannot enqueue data: queue is full");

        buffer[last++] = data;
        if(last == size)
            last = 0;
        counter++;

        pthread_cond_signal(&cvxQueue);
        pthread_mutex_unlock(&mtxQueue);
    }
    void
    pop(T &data) {
        pthread_mutex_lock(&mtxQueue);

        if(counter == 0)
            pthread_cond_wait(&cvxQueue,&mtxQueue);

        data = buffer[first++];
        if(first == size)
            first = 0;
        counter--;

        pthread_mutex_unlock(&mtxQueue);
    }
    void
    show() {
        uint    i,n;

        std::cout << '<';

        i = first;
        n = counter;
        while(n-- > 0) {
            std::cout << buffer[i++];
            if(i == size)
                i = 0;
            if(n > 0)
                std::cout << ',';
        }
        std::cout << '>' << std::endl;
    }
};

#endif	// __queue_hpp
