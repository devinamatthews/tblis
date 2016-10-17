#ifndef _TBLIS_KERNELS_1M_REDUCE_HPP_
#define _TBLIS_KERNELS_1M_REDUCE_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

namespace tblis
{

template <typename T>
using matrix_reduce_ker_t =
    void (*)(communicator& comm,
             len_type m, len_type n,
             const T* TBLIS_RESTRICT A,
             stride_type rs_A, stride_type cs_A,
             T& TBLIS_RESTRICT norm);

template <typename T>
void matrix_reduce_ker_def(communicator& comm,
                           len_type m, len_type n,
                           const T* TBLIS_RESTRICT A,
                           stride_type rs_A, stride_type cs_A,
                           T& TBLIS_RESTRICT norm)
{
    T subnrm = T();

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) = comm.distribute_over_threads_2d(m, n);

    if (rs_A == 1)
    {
        for (len_type j = n_min;j < n_max;j++)
        {
            for (len_type i = m_min;i < m_max;i++)
            {
                subnrm += norm2(A[i+cs_A*j]);
            }
        }
    }
    else if (cs_A == 1)
    {
        for (len_type i = m_min;i < m_max;i++)
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                subnrm += norm2(A[rs_A*i+j]);
            }
        }
    }
    else
    {
        for (len_type i = m_min;i < m_max;i++)
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                subnrm += norm2(A[rs_A*i+cs_A*j]);
            }
        }
    }

    norm = sqrt(real(reduce(comm, subnrm)));
}

}

#endif
