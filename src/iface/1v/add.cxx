#include "add.h"

#include "util/macros.h"
#include "internal/1v/add.hpp"
#include "internal/1v/scale.hpp"
#include "internal/1v/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_vector_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, tblis_vector* B)
{
    TBLIS_ASSERT(A->n == B->n);
    TBLIS_ASSERT(A->type == B->type);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        if (A->alpha<T>() == T(0))
        {
            if (B->alpha<T>() == T(0))
            {
                parallelize_if(internal::set<T>, comm, get_config(cfg), A->n,
                               T(0), static_cast<T*>(B->data), B->inc);
            }
            else
            {
                parallelize_if(internal::scale<T>, comm, get_config(cfg), A->n,
                               B->alpha<T>(), B->conj, static_cast<T*>(B->data), B->inc);
            }
        }
        else
        {
            parallelize_if(internal::add<T>, comm, get_config(cfg), A->n,
                           A->alpha<T>(), A->conj, static_cast<const T*>(A->data), A->inc,
                           B->alpha<T>(), B->conj,       static_cast<T*>(B->data), B->inc);
        }

        B->alpha<T>() = T(1);
        B->conj = false;
    })
}

}

}
