#include "mult.h"

#include "util/macros.h"
#include "internal/1m/scale.hpp"
#include "internal/1m/set.hpp"
#include "internal/3m/mult.hpp"

namespace tblis
{

extern "C"
{

void tblis_matrix_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_matrix* A, const tblis_matrix* B,
                       tblis_matrix* C)
{
    TBLIS_ASSERT(A->m == C->m);
    TBLIS_ASSERT(B->n == C->n);
    TBLIS_ASSERT(A->n == B->m);
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);

    TBLIS_ASSERT(!A->conj);
    TBLIS_ASSERT(!B->conj);
    TBLIS_ASSERT(!C->conj);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T alpha = A->alpha<T>()*B->alpha<T>();
        T beta = C->alpha<T>();

        if (alpha == T(0))
        {
            if (beta == T(0))
            {
                parallelize_if(internal::set<T>, comm, get_config(cfg), C->m, C->n,
                               T(0), static_cast<T*>(C->data), C->rs, C->cs);
            }
            else
            {
                parallelize_if(internal::scale<T>, comm, get_config(cfg), C->m, C->n,
                               beta, C->conj, static_cast<T*>(C->data), C->rs, C->cs);
            }
        }
        else
        {
            parallelize_if(internal::mult<T>, comm, get_config(cfg),
                           C->m, C->n, A->n,
                           alpha, A->conj, static_cast<const T*>(A->data), A->rs, A->cs,
                                  B->conj, static_cast<const T*>(B->data), B->rs, B->cs,
                            beta, C->conj,       static_cast<T*>(C->data), C->rs, C->cs);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

}

}
