#include "set.h"

#include "configs/configs.hpp"

namespace tblis
{

void set_int(const communicator& comm, const config& cfg,
             const tblis_scalar& alpha, tblis_matrix& A)
{
    TBLIS_ASSERT(alpha.type == A.type);

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(A.m, A.n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        if (A.rs < A.cs)
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                cfg.set_ukr.call<T>(m_max-m_min,
                    alpha.get<T>(), (T*)A.data+ m_min*A.rs + j*A.cs, A.rs);
            }
        }
        else
        {
            for (len_type i = m_min;i < m_max;i++)
            {
                cfg.set_ukr.call<T>(n_max-n_min,
                    alpha.get<T>(), (T*)A.data+ i*A.rs + n_min*A.cs, A.cs);
            }
        }

        A.alpha<T>() = T(1);
        A.conj = false;
    })
}

extern "C"
{

void tblis_matrix_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_matrix* A)
{
    parallelize_if(set_int, comm, get_config(cfg),
                   *alpha, *A);
}

}

}
