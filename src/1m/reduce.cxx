#include "reduce.h"

#include "configs/configs.hpp"

namespace tblis
{

extern "C"
{

void tblis_matrix_reduce_int(const communicator& comm, const config& cfg,
                             reduce_t op, const tblis_matrix& A,
                             tblis_scalar& result, len_type& idx)
{
    TBLIS_ASSERT(A.type == result.type);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(A.m, A.n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        reduce_init(op, result.get<T>(), idx);

        if (A.rs < A.cs)
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                int old_idx = idx;
                idx = -1;

                cfg.reduce_ukr.call<T>(op, m_max-m_min,
                    (const T*)A.data+ m_min*A.rs + j*A.cs, A.rs, result.get<T>(), idx);

                if (idx != -1) idx += j*A.cs;
                else idx = old_idx;
            }
        }
        else
        {
            for (len_type i = m_min;i < m_max;i++)
            {
                int old_idx = idx;
                idx = -1;

                cfg.reduce_ukr.call<T>(op, n_max-n_min,
                    (const T*)A.data+ i*A.rs + n_min*A.cs, A.cs, result.get<T>(), idx);

                if (idx != -1) idx += i*A.rs;
                else idx = old_idx;
            }
        }

        reduce(comm, op, result.get<T>(), idx);
    })
}

void tblis_matrix_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_matrix* A,
                         tblis_scalar* result, len_type* idx)
{
    parallelize_if(tblis_matrix_reduce_int, comm, get_config(cfg),
                   op, *A, *result, *idx);
}

}

}
