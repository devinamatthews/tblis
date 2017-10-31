#include "reduce.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op,
            len_type m, len_type n, const T* A, stride_type rs_A, stride_type cs_A,
            T& result, len_type& idx)
{
    if (rs_A > cs_A)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
    }

    atomic_reducer<T> local_result;
    reduce_init(op, local_result);

    comm.distribute_over_threads(m, n,
    [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
    {
        T micro_result;
        len_type micro_idx;
        reduce_init(op, micro_result, micro_idx);

        for (len_type j = n_min;j < n_max;j++)
        {
            auto old_idx = micro_idx;
            micro_idx = -1;

            cfg.reduce_ukr.call<T>(op, m_max-m_min,
                                   A + m_min*rs_A + j*cs_A, rs_A, micro_result, micro_idx);

            if (micro_idx != -1) micro_idx += m_min*rs_A + j*cs_A;
            else micro_idx = old_idx;
        }

        atomic_reduce(op, local_result, micro_result, micro_idx);
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
                     len_type m, len_type n, const T* A, stride_type rs_A, stride_type cs_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
