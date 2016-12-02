#include "set.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void set(const communicator& comm, const config& cfg, len_type m, len_type n,
         T alpha, T* A, stride_type rs_A, stride_type cs_A)
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

    for (len_type j = n_min;j < n_max;j++)
    {
        cfg.set_ukr.call<T>(m_max-m_min,
                            alpha, A + m_min*rs_A + j*cs_A, rs_A);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void set(const communicator& comm, const config& cfg, len_type m, len_type n, \
                  T alpha, T* A, stride_type rs_A, stride_type cs_A);
#include "configs/foreach_type.h"

}
}
