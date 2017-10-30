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
    if (rs_B > cs_B)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
        std::swap(rs_B, cs_B);
    }

    atomic_accumulator<T> local_result;

    comm.distribute_over_threads(tci::range(m).chunk(1000),
                                 tci::range(n).chunk(1000/m),
    [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
    {
        T micro_result = T();

        for (len_type j = n_min;j < n_max;j++)
        {
            cfg.dot_ukr.call<T>(m_max-m_min,
                                conj_A, A + m_min*rs_A + j*cs_A, rs_A,
                                conj_B, B + m_min*rs_B + j*cs_B, rs_B, micro_result);
        }

        local_result += micro_result;
    });

    reduce(comm, local_result);
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
