#include "dot.h"

#include "configs/configs.hpp"

namespace tblis
{

void dot_int(const communicator& comm, const config& cfg,
             const tblis_matrix& A, const tblis_matrix& B,
             tblis_scalar& result)
{
    TBLIS_ASSERT(A.m == B.m);
    TBLIS_ASSERT(A.n == B.n);
    TBLIS_ASSERT(A.type == B.type);
    TBLIS_ASSERT(A.type == result.type);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(A.m, A.n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        result.get<T>() = T();

        if (A.rs < A.cs)
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                cfg.dot_ukr.call<T>(m_max-m_min,
                    A.conj, (const T*)A.data+ m_min*A.rs + j*A.cs, A.rs,
                    B.conj, (const T*)B.data+ m_min*B.rs + j*B.cs, B.rs, result.get<T>());
            }
        }
        else
        {
            for (len_type i = m_min;i < m_max;i++)
            {
                cfg.dot_ukr.call<T>(n_max-n_min,
                    A.conj, (const T*)A.data+ i*A.rs + n_min*A.cs, A.cs,
                    B.conj, (const T*)B.data+ i*B.rs + n_min*B.cs, B.cs, result.get<T>());
            }
        }

        len_type dummy;
        reduce(comm, REDUCE_SUM, result.get<T>(), dummy);
    })
}

extern "C"
{

void tblis_matrix_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_matrix* A, const tblis_matrix* B,
                      tblis_scalar* result)
{
    parallelize_if(dot_int, comm, get_config(cfg),
                   *A, *B, *result);
}

}

}
