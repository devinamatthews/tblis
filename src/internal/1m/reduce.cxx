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
    const bool trans = rs_A > cs_A;

    if (trans)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
    }

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    reduce_init(op, result, idx);

    for (len_type j = n_min;j < n_max;j++)
    {
        int old_idx = idx;
        idx = -1;

        cfg.reduce_ukr.call<T>(op, m_max-m_min,
                               A + m_min*rs_A + j*cs_A, rs_A, result, idx);

        if (idx != -1) idx += j*cs_A;
        else idx = old_idx;
    }

    reduce(comm, op, result, idx);
}

#define FOREACH_TYPE(T) \
template void reduce(const communicator& comm, const config& cfg, reduce_t op, \
                     len_type m, len_type n, const T* A, stride_type rs_A, stride_type cs_A, \
                     T& result, len_type& idx);
#include "configs/foreach_type.h"

}
}
