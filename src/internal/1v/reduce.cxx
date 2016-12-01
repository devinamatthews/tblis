#include "reduce.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op, len_type n,
            const T* A, stride_type inc_A, T& result, len_type& idx)
{
    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    T local_result;
    len_type local_idx;
    reduce_init(op, local_result, local_idx);

    cfg.reduce_ukr.call<T>(op, n_max-n_min,
                           A + n_min*inc_A, inc_A, local_result, local_idx);

    reduce(comm, op, local_result, local_idx);

    if (comm.master())
    {
        result = local_result;
        idx = local_idx;
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
