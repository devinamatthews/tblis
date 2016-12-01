#include "dot.h"

#include "util/macros.h"
#include "internal/1v/dot.hpp"

namespace tblis
{

extern "C"
{

void tblis_vector_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, const tblis_vector* B,
                      tblis_scalar* result)
{
    TBLIS_ASSERT(A->n == B->n);
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == result->type);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(internal::dot<T>, comm, get_config(cfg), A->n,
                       A->conj, static_cast<const T*>(A->data), A->inc,
                       B->conj, static_cast<const T*>(B->data), B->inc, result->get<T>());

        result->get<T>() *= A->alpha<T>()*B->alpha<T>();
    })
}

}

}
