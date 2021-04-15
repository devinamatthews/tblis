#ifndef TBLIS_BASE_THREAD_H
#define TBLIS_BASE_THREAD_H

#include <tci.h>

#include <tblis/base/macros.h>
#include <tblis/base/types.h>
#include <tblis/base/env.h>

typedef tci_comm tblis_comm;
extern const tblis_comm* const tblis_single;

#if TBLIS_ENABLE_CXX

TBLIS_BEGIN_NAMESPACE

using tci::communicator;
using tci::parallelize;
using tci::partition_2x2;

extern communicator single;

template <typename Func, typename... Args>
void parallelize_if(const Func& f, const tblis_comm* _comm, Args&&... args)
{
    if (_comm)
    {
        f(*reinterpret_cast<const communicator*>(_comm), args...);
    }
    else
    {
        parallelize
        (
            [&,f](const communicator& comm) mutable
            {
                f(comm, args...);
                comm.barrier();
            },
            tblis_get_num_threads()
        );
    }
}

TBLIS_END_NAMESPACE

#endif //TBLIS_ENABLE_CXX

#endif //TBLIS_BASE_THREAD_H
