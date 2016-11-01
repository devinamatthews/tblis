#include "reduce.h"

#include "configs/configs.hpp"

namespace tblis
{

void reduce_int(const communicator& comm, const config& cfg,
                reduce_t op, const tblis_vector& A,
                tblis_scalar& result, len_type& idx)
{
    TBLIS_ASSERT(A.type == result.type);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(A.n);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        reduce_init(op, result.get<T>(), idx);

        cfg.reduce_ukr.call<T>(op, n_max-n_min,
            (const T*)A.data + n_min*A.inc, A.inc, result.get<T>(), idx);

        reduce(comm, op, result.get<T>(), idx);
    })
}

extern "C"
{

void tblis_vector_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_vector* A,
                         tblis_scalar* result, len_type* idx)
{
    parallelize_if(reduce_int, comm, get_config(cfg),
                   op, *A, *result, *idx);
}

}

}
