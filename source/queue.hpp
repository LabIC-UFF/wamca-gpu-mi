/**
 * @file	queue.hpp
 *
 * @brief	Implements a circular queue.
 *
 * @author	Eyder Rios
 * @date    2015-05-28
 */

#ifndef __queue_hpp
#define __queue_hpp

#include <iostream>
#include "except.h"


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
// ##                                 CLASS Queue                                ## //
// ##                                                                            ## //
// ################################################################################ //

template<typename T, int size>
class Queue
{
private:
    uint    counter;
    uint    first;
    uint    last;
    T       buffer[size];

public:
    Queue() {
        clear();
    }
    inline
    bool
    count() {
        return counter;
    }
    inline
    bool
    empty() {
        return counter == 0;
    }
    inline
    bool
    notEmpty() {
        return counter > 0;
    }
    inline
    bool
    full() {
        return counter == size;
    }
    inline
    void
    clear() {
        counter= 0;
        first  = 0;
        last   = 0;
    }
    void
    push(const T &data) {
        if(counter == size)
            EXCEPTION("Cannot enqueue data: queue is full");

        buffer[last++] = data;
        if(last == size)
            last = 0;
        counter++;
    }
    void
    pop(T &data) {
        if(counter == 0)
            EXCEPTION("Cannot dequeue data: queue is empty");

        data = buffer[first++];
        if(first == size)
            first = 0;
        counter--;
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
