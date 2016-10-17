#ifndef _TBLIS_THREAD_HPP_
#define _TBLIS_THREAD_HPP_

#include "external/tci/src/tci.h"

#ifdef __cplusplus

namespace tblis
{

using namespace tci;

template <typename T>
T reduce(communicator& comm, T value)
{
    if (comm.num_threads() == 1) return value;

    T* vals;
    std::vector<T> val_buffer;

    if (comm.master())
    {
        val_buffer.resize(comm.num_threads());
        vals = val_buffer.data();
    }

    comm.broadcast_nowait(vals);

    vals[comm.thread_num()] = value;

    comm.barrier();

    if (comm.master())
    {
        for (int i = 1;i < comm.num_threads();i++)
        {
            vals[0] += vals[i];
        }
    }

    comm.barrier();

    value = vals[0];

    comm.barrier();

    return value;
}

}

#endif

#endif
