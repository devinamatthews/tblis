#include "dot.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg, len_type m, len_type n,
         bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
         bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, T& result)
{
    const bool trans = rs_B > cs_B;

    if (trans)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
        std::swap(rs_B, cs_B);
    }

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    T local_result = T();

    for (len_type j = n_min;j < n_max;j++)
    {
        cfg.dot_ukr.call<T>(m_max-m_min,
                            conj_A, A + m_min*rs_A + j*cs_A, rs_A,
                            conj_B, B + m_min*rs_B + j*cs_B, rs_B, result);
    }

    len_type dummy;
    reduce(comm, REDUCE_SUM, local_result, dummy);
    if (comm.master()) result = local_result;

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, len_type m, len_type n, \
                  bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                  bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, T& result);
#include "configs/foreach_type.h"

}
}
