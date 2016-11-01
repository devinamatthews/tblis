#include "scale.h"

#include "configs/configs.hpp"

namespace tblis
{

void scale_int(const communicator& comm, const config& cfg,
               tblis_vector& A)
{
    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(A.n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        cfg.scale_ukr.call<T>(n_max-n_min,
            A.alpha<T>(), A.conj, (T*)A.data + n_min*A.inc, A.inc);

        if (comm.master())
        {
            A.alpha<T>() = T(1);
            A.conj = false;
        }
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
