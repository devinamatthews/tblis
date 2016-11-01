#include "add.h"

#include "configs/configs.hpp"

namespace tblis
{

void add_int(const communicator& comm, const config& cfg,
             const tblis_vector& A, tblis_vector& B)
{
    TBLIS_ASSERT(A.n == B.n);
    TBLIS_ASSERT(A.type == B.type);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(A.n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        if (A.alpha<T>() == T(0))
        {
            if (B.alpha<T>() == T(0))
            {
                cfg.set_ukr.call<T>(n_max-n_min, T(0), (T*)B.data + n_min*B.inc, B.inc);
            }
            else
            {
                cfg.scale_ukr.call<T>(n_max-n_min,
                    B.alpha<T>(), B.conj, (T*)B.data + n_min*B.inc, B.inc);
            }
        }
        else
        {
            if (B.alpha<T>() == T(0))
            {
                cfg.copy_ukr.call<T>(n_max-n_min,
                    A.alpha<T>(), A.conj, (const T*)A.data + n_min*A.inc, A.inc,
                                                (T*)B.data + n_min*B.inc, B.inc);
            }
            else
            {
                cfg.add_ukr.call<T>(n_max-n_min,
                    A.alpha<T>(), A.conj, (const T*)A.data + n_min*A.inc, A.inc,
                    B.alpha<T>(), B.conj,       (T*)B.data + n_min*B.inc, B.inc);
            }
        }

        if (comm.master())
        {
            B.alpha<T>() = T(1);
            B.conj = false;
        }
    })
}

extern "C"
{

void tblis_vector_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, tblis_vector* B)
{
    parallelize_if(add_int, comm, get_config(cfg),
                   *A, *B);
}

}

}
