#include "dot.h"

#include "configs/configs.hpp"

namespace tblis
{

void dot_int(const communicator& comm, const config& cfg,
             const tblis_vector& A, const tblis_vector& B,
             tblis_scalar& result)
{
    TBLIS_ASSERT(A.n == B.n);
    TBLIS_ASSERT(A.type == B.type);
    TBLIS_ASSERT(A.type == result.type);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        result.get<T>() = T();

        cfg.dot_ukr.call<T>(comm, A.n,
            A.conj, (const T*)A.data, A.inc,
            B.conj, (const T*)B.data, B.inc, result.get<T>());

        len_type dummy;
        reduce(comm, REDUCE_SUM, result.get<T>(), dummy);
    })
}

extern "C"
{

void tblis_vector_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, const tblis_vector* B,
                      tblis_scalar* result)
{
    parallelize_if(dot_int, comm, get_config(cfg),
                   *A, *B, *result);
}

}

}
