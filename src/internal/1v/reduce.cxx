#include "reduce.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op, len_type n,
            const T* A, stride_type inc_A, T& result, len_type& idx)
{
    atomic_reducer<T> local_result{reduce_init<T>(op)};

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        T micro_result;
        len_type micro_idx;
        reduce_init(op, micro_result, micro_idx);

        cfg.reduce_ukr.call<T>(op, n_max-n_min,
                               A + n_min*inc_A, inc_A, micro_result, micro_idx);

        atomic_reduce(op, local_result, micro_result, micro_idx + n_min*inc_A);
    });

    reduce(comm, op, local_result);

    if (comm.master())
    {
        result = local_result.load().first;
        idx = local_result.load().second;
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, const config& cfg, reduce_t op, \
                     len_type n, const T* A, stride_type inc_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
