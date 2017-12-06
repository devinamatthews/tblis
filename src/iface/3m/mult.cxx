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

        if (C->m != 0 && C->n != 0)
        {
            parallelize_if(
            [&](const communicator& comm)
            {
                if (alpha == T(0) || A->n == 0)
                {
                    if (beta == T(0))
                    {
                        internal::set<T>(comm, get_config(cfg), C->m, C->n,
                                         T(0), static_cast<T*>(C->data), C->rs, C->cs);
                    }
                    else if (beta != T(1) || (is_complex<T>::value && C->conj))
                    {
                        internal::scale<T>(comm, get_config(cfg), C->m, C->n,
                                           beta, C->conj,
                                           static_cast<T*>(C->data), C->rs, C->cs);
                    }
                }
                else
                {
                    internal::mult<T>(comm, get_config(cfg),
                                      C->m, C->n, A->n,
                                      alpha, A->conj, static_cast<const T*>(A->data), A->rs, A->cs,
                                             B->conj, static_cast<const T*>(B->data), B->rs, B->cs,
                                       beta, C->conj,       static_cast<T*>(C->data), C->rs, C->cs);
                }
            }, comm);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

void tblis_matrix_mult_diag(const tblis_comm* comm, const tblis_config* cfg,
                            const tblis_matrix* A,
                            const tblis_vector* D,
                            const tblis_matrix* B,
                            tblis_matrix* C)
{
    TBLIS_ASSERT(A->m == C->m);
    TBLIS_ASSERT(B->n == C->n);
    TBLIS_ASSERT(A->n == B->m);
    TBLIS_ASSERT(A->n == D->n);
    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);
    TBLIS_ASSERT(A->type == D->type);

    TBLIS_ASSERT(!A->conj);
    TBLIS_ASSERT(!B->conj);
    TBLIS_ASSERT(!C->conj);
    TBLIS_ASSERT(!D->conj);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T alpha = A->alpha<T>()*B->alpha<T>();
        T beta = C->alpha<T>();

        if (C->m != 0 && C->n != 0)
        {
            parallelize_if(
            [&](const communicator& comm)
            {
                if (alpha == T(0) || A->n == 0)
                {
                    if (beta == T(0))
                    {
                        internal::set<T>(comm, get_config(cfg), C->m, C->n,
                                         T(0), static_cast<T*>(C->data), C->rs, C->cs);
                    }
                    else if (beta != T(1) || (is_complex<T>::value && C->conj))
                    {
                        internal::scale<T>(comm, get_config(cfg), C->m, C->n,
                                           beta, C->conj,
                                           static_cast<T*>(C->data), C->rs, C->cs);
                    }
                }
                else
                {
                    internal::mult<T>(comm, get_config(cfg),
                                      C->m, C->n, A->n,
                                      alpha, A->conj, static_cast<const T*>(A->data), A->rs, A->cs,
                                             D->conj, static_cast<const T*>(D->data), D->inc,
                                             B->conj, static_cast<const T*>(B->data), B->rs, B->cs,
                                       beta, C->conj,       static_cast<T*>(C->data), C->rs, C->cs);
                }
            }, comm);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

}

}
