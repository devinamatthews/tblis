#include "add.h"

#include "util/macros.h"
#include "internal/1m/add.hpp"
#include "internal/1m/scale.hpp"
#include "internal/1m/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_matrix_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_matrix* A, tblis_matrix* B)
{
    TBLIS_ASSERT(A->m == B->m);
    TBLIS_ASSERT(A->n == B->n);
    TBLIS_ASSERT(A->type == B->type);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        if (A->alpha<T>() == T(0))
        {
            if (B->alpha<T>() == T(0))
            {
                parallelize_if(internal::set<T>, comm, get_config(cfg), A->m, A->n,
                               T(0), static_cast<T*>(B->data), B->rs, B->cs);
            }
            else
            {
                parallelize_if(internal::scale<T>, comm, get_config(cfg), A->m, A->n,
                               B->alpha<T>(), B->conj, static_cast<T*>(B->data), B->rs, B->cs);
            }
        }
        else
        {
            parallelize_if(internal::add<T>, comm, get_config(cfg), A->m, A->n,
                           A->alpha<T>(), A->conj, static_cast<const T*>(A->data), A->rs, A->cs,
                           B->alpha<T>(), B->conj,       static_cast<T*>(B->data), B->rs, B->cs);
        }

        B->alpha<T>() = T(1);
        B->conj = false;
    })
}

}

}
