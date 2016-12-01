#ifndef _TBLIS_THREAD_HPP_
#define _TBLIS_THREAD_HPP_

#include "tci.h"
#include "basic_types.h"

typedef tci_comm_t tblis_comm;
extern const tblis_comm* const tblis_single;

#ifdef __cplusplus
extern "C"
{
#endif

unsigned tblis_get_num_threads();

void tblis_set_num_threads(unsigned num_threads);

#ifdef __cplusplus
}

#include <vector>
#include <utility>

namespace tblis
{

using namespace tci;

template <typename T>
void reduce_init(reduce_t op, T& value, len_type& idx)
{
    typedef std::numeric_limits<real_type_t<T>> limits;

    switch (op)
    {
        case REDUCE_SUM:
        case REDUCE_SUM_ABS:
        case REDUCE_MAX_ABS:
        case REDUCE_NORM_2:
            value = T();
            break;
        case REDUCE_MAX:
            value = limits::min();
            break;
        case REDUCE_MIN:
        case REDUCE_MIN_ABS:
            value = limits::max();
            break;
    }

    idx = -1;
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, T& value, len_type& idx)
{
    if (comm.num_threads() == 1)
    {
        if (op == REDUCE_NORM_2) value = sqrt(value);
        return;
    }

    std::pair<T,len_type>* vals;
    std::vector<std::pair<T,len_type>> val_buffer;

    if (comm.master())
    {
        val_buffer.resize(comm.num_threads());
        vals = val_buffer.data();
    }

    comm.broadcast_nowait(vals);

    vals[comm.thread_num()] = {value, idx};

    comm.barrier();

    if (comm.master())
    {
        if (op == REDUCE_SUM)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                vals[0].first += vals[i].first;
            }
        }
        else if (op == REDUCE_SUM_ABS)
        {
            vals[0].first = std::abs(vals[0].first);
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                vals[0].first += std::abs(vals[i].first);
            }
        }
        else if (op == REDUCE_MAX)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (vals[i].first > vals[0].first) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_MAX_ABS)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (std::abs(vals[i].first) >
                    std::abs(vals[0].first)) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_MIN)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (vals[i].first < vals[0].first) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_MIN_ABS)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                if (std::abs(vals[i].first) <
                    std::abs(vals[0].first)) vals[0] = vals[i];
            }
        }
        else if (op == REDUCE_NORM_2)
        {
            for (unsigned i = 1;i < comm.num_threads();i++)
            {
                vals[0].first += vals[i].first;
            }
            vals[0].first = std::sqrt(vals[0].first);
        }
    }

    comm.barrier();

    value = vals[0].first;
    idx = vals[0].second;

    comm.barrier();
}

template <typename Func, typename... Args>
void parallelize_if(Func f, const tblis_comm* _comm, Args&&... args)
{
    if (_comm)
    {
        f(*reinterpret_cast<const communicator*>(_comm), args...);
    }
    else
    {
        parallelize
        (
            [&](const communicator& comm)
            {
                f(comm, args...);
                comm.barrier();
            },
            tblis_get_num_threads()
        );
    }
}

}

#endif

#endif
