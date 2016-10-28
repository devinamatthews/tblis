#include "add.h"

#include "configs/configs.hpp"

namespace tblis
{

void add_int(const communicator& comm, const config& cfg,
             const tblis_vector& A, tblis_vector& B)
{
    TBLIS_ASSERT(A.n == B.n);
    TBLIS_ASSERT(A.type == B.type);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        if (A.alpha<T>() == T(0))
        {
            if (B.alpha<T>() == T(0))
            {
                cfg.set_ukr.call<T>(comm, A.n, T(0), (T*)B.data, B.inc);
            }
            else
            {
                cfg.scale_ukr.call<T>(comm, A.n,
                    B.alpha<T>(), B.conj, (T*)B.data, B.inc);
            }
        }
        else
        {
            if (B.alpha<T>() == T(0))
            {
                cfg.copy_ukr.call<T>(comm, A.n,
                    A.alpha<T>(), A.conj, (const T*)A.data, A.inc,
                                                  (T*)B.data, B.inc);
            }
            else
            {
                cfg.add_ukr.call<T>(comm, A.n,
                    A.alpha<T>(), A.conj, (const T*)A.data, A.inc,
                    B.alpha<T>(), B.conj,       (T*)B.data, B.inc);
            }
        }

        B.alpha<T>() = T(1);
        B.conj = false;
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
