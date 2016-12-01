#include "set.h"

#include "util/macros.h"
#include "internal/1v/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_vector_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_vector* A)
{
    TBLIS_ASSERT(alpha->type == A->type);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(internal::set<T>, comm, get_config(cfg), A->n,
                       alpha->get<T>(), static_cast<T*>(A->data), A->inc);

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

}
