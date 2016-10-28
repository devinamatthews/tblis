#include "scale.h"

#include "configs/configs.hpp"

namespace tblis
{

void scale_int(const communicator& comm, const config& cfg,
               tblis_vector& A)
{
    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        cfg.scale_ukr.call<T>(comm, A.n,
            A.alpha<T>(), A.conj, (T*)A.data, A.inc);

        A.alpha<T>() = T(1);
        A.conj = false;
    })
}

extern "C"
{

void tblis_vector_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_vector* A)
{
    parallelize_if(scale_int, comm, get_config(cfg),
                   *A);
}

}

}
