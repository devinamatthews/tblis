#include "set.h"

#include "configs/configs.hpp"

namespace tblis
{

void set_int(const communicator& comm, const config& cfg,
             const tblis_scalar& alpha, tblis_vector& A)
{
    TBLIS_ASSERT(alpha.type == A.type);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        cfg.set_ukr.call<T>(comm, A.n,
            alpha.get<T>(), (T*)A.data, A.inc);

        A.alpha<T>() = T(1);
        A.conj = false;
    })
}

extern "C"
{

void tblis_vector_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_vector* A)
{
    parallelize_if(set_int, comm, get_config(cfg),
                   *alpha, *A);
}

}

}
