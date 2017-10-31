#include "shift.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void shift(const communicator& comm, const config& cfg, len_type m, len_type n,
           T alpha, T beta, bool conj_A, T* A, stride_type rs_A, stride_type cs_A)
{
    if (rs_A > cs_A)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
    }

    comm.distribute_over_threads(m, n,
    [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
    {
        for (len_type j = n_min;j < n_max;j++)
        {
            cfg.shift_ukr.call<T>(m_max-m_min,
                                  alpha, beta, conj_A, A+ m_min*rs_A + j*cs_A, rs_A);
        }
    });

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void shift(const communicator& comm, const config& cfg, len_type m, len_type n, \
                    T alpha, T beta, bool conj_A, T* A, stride_type rs_A, stride_type cs_A);
#include "configs/foreach_type.h"

}
}
