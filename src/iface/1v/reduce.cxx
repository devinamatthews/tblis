#include "reduce.h"

#include "util/macros.h"
#include "internal/1v/reduce.hpp"

namespace tblis
{

extern "C"
{

void tblis_vector_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_vector* A,
                         tblis_scalar* result, len_type* idx)
{
    TBLIS_ASSERT(A->type == result->type);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        if (A->alpha<T>() < T(0))
        {
            if (op == REDUCE_MIN) op = REDUCE_MAX;
            else if (op == REDUCE_MAX) op = REDUCE_MIN;
        }

        parallelize_if(internal::reduce<T>, comm, get_config(cfg), op,  A->n,
                       static_cast<const T*>(A->data), A->inc, result->get<T>(), *idx);

        if (A->conj)
        {
            result->get<T>() = conj(result->get<T>());
        }

        if (op == REDUCE_SUM || op == REDUCE_SUM_ABS || op == REDUCE_NORM_2)
        {
            result->get<T>() *= A->alpha<T>();
        }
    })
}

}

}
