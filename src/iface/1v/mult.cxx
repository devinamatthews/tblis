#include "mult.h"

#include "util/macros.h"
#include "internal/1v/mult.hpp"
#include "internal/1v/scale.hpp"
#include "internal/1v/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_vector_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_vector* A, const tblis_vector* B, tblis_vector* C)
{
    TBLIS_ASSERT(A->n == B->n);
    TBLIS_ASSERT(A->n == C->n);
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(
        [&](const communicator& comm)
        {
            T alpha = A->alpha<T>()*B->alpha<T>();

            if (alpha == T(0))
            {
                if (C->alpha<T>() == T(0))
                {
                    internal::set<T>(comm, get_config(cfg), A->n,
                                     T(0), static_cast<T*>(C->data), C->inc);
                }
                else if (C->alpha<T>() != T(1) || (is_complex<T>::value && C->conj))
                {
                    internal::scale<T>(comm, get_config(cfg), A->n,
                                       C->alpha<T>(), C->conj,
                                       static_cast<T*>(C->data), C->inc);
                }
            }
            else
            {
                internal::mult<T>(comm, get_config(cfg), A->n,
                                  alpha, A->conj, static_cast<const T*>(A->data), A->inc,
                                         B->conj, static_cast<const T*>(B->data), B->inc,
                                  C->alpha<T>(), C->conj, static_cast<T*>(C->data), C->inc);
            }
        }, comm);

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

}

}
