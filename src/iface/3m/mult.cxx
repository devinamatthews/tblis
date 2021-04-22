#include "mult.h"

#include "util/macros.h"
#include "internal/1v/add.hpp"
#include "internal/1v/dot.hpp"
#include "internal/1v/scale.hpp"
#include "internal/1v/set.hpp"
#include "internal/1m/scale.hpp"
#include "internal/1m/set.hpp"
#include "internal/2m/mult.hpp"
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
                        if (C->m == 1 && C->n == 1)
                        {
                            if (comm.master())
                                *((T*)C->data) = T(0);
                        }
                        else if (C->m == 1)
                        {
                            internal::set<T>(comm, get_config(cfg), C->n,
                                             T(0), ((T*)C->data), C->cs);
                        }
                        else if (C->n== 1)
                        {
                            internal::set<T>(comm, get_config(cfg), C->m,
                                             T(0), ((T*)C->data), C->rs);
                        }
                        else
                        {
                            internal::set<T>(comm, get_config(cfg), C->m, C->n,
                                             T(0), ((T*)C->data), C->rs, C->cs);
                        }
                    }
                    else if (beta != T(1) || (is_complex<T>::value && C->conj))
                    {
                        if (C->m == 1 && C->n == 1)
                        {
                            if (comm.master())
                                *((T*)C->data) = beta*(C->conj ? conj(*((T*)C->data)) : *((T*)C->data));
                        }
                        else if (C->m == 1)
                        {
                            internal::scale<T>(comm, get_config(cfg), C->n,
                                               beta, C->conj, ((T*)C->data), C->cs);
                        }
                        else if (C->n == 1)
                        {
                            internal::scale<T>(comm, get_config(cfg), C->m,
                                               beta, C->conj, ((T*)C->data), C->rs);
                        }
                        else
                        {
                            internal::scale<T>(comm, get_config(cfg), C->m, C->n,
                                               beta, C->conj, ((T*)C->data), C->rs, C->cs);
                        }
                    }
                }
                else
                {
                    if (A->n == 1)
                    {
                        if (C->m == 1 && C->n == 1)
                        {
                            if (comm.master())
                            {
                                if (beta == T(0))
                                {
                                    *((T*)C->data) = alpha*(A->conj ? conj(*((T*)A->data)) : *((T*)A->data))*
                                                           (B->conj ? conj(*((T*)B->data)) : *((T*)B->data));
                                }
                                else
                                {
                                    *((T*)C->data) = alpha*(A->conj ? conj(*((T*)A->data)) : *((T*)A->data))*
                                                           (B->conj ? conj(*((T*)B->data)) : *((T*)B->data)) +
                                                      beta*(C->conj ? conj(*((T*)C->data)) : *((T*)C->data));
                                }
                            }
                        }
                        else if (C->m == 1)
                        {
                            internal::add<T>(comm, get_config(cfg), C->n,
                                             alpha*(A->conj ? conj(*((T*)A->data)) : *((T*)A->data)),
                                             B->conj, ((T*)B->data), B->cs,
                                             beta, C->conj, ((T*)C->data), C->cs);
                        }
                        else if (C->n == 1)
                        {
                            internal::add<T>(comm, get_config(cfg), C->m,
                                             alpha*(B->conj ? conj(*((T*)B->data)) : *((T*)B->data)),
                                             A->conj, ((T*)A->data), A->rs,
                                             beta, C->conj, ((T*)C->data), C->rs);
                        }
                        else
                        {
                            internal::mult<T>(comm, get_config(cfg),
                                              C->m, C->n,
                                              alpha, A->conj, ((T*)A->data), A->rs,
                                                     B->conj, ((T*)B->data), B->cs,
                                               beta, C->conj, ((T*)C->data), C->rs, C->cs);
                        }
                    }
                    else
                    {
                        if (C->m == 1 && C->n == 1)
                        {
                            T result = T(0);
                            internal::dot<T>(comm, get_config(cfg), A->n,
                                             A->conj, ((T*)A->data), A->cs,
                                             B->conj, ((T*)B->data), B->rs,
                                             result);

                            if (comm.master())
                            {
                                if (beta == T(0))
                                {
                                    *((T*)C->data) = alpha*result;
                                }
                                else
                                {
                                    *((T*)C->data) = alpha*result +
                                        beta*(C->conj ? conj(*((T*)C->data)) : *((T*)C->data));
                                }
                            }
                        }
                        else if (C->m == 1)
                        {
                            internal::mult<T>(comm, get_config(cfg),
                                              B->n, B->m,
                                              alpha, B->conj, ((T*)B->data), B->cs, B->rs,
                                                     A->conj, ((T*)A->data), A->cs,
                                               beta, C->conj, ((T*)C->data), C->cs);
                        }
                        else if (C->n == 1)
                        {
                            internal::mult<T>(comm, get_config(cfg),
                                              A->m, A->n,
                                              alpha, A->conj, ((T*)A->data), A->rs, A->cs,
                                                     B->conj, ((T*)B->data), B->rs,
                                               beta, C->conj, ((T*)C->data), C->rs);
                        }
                        else
                        {
                            internal::mult<T>(comm, get_config(cfg),
                                              C->m, C->n, A->n,
                                              alpha, A->conj, ((T*)A->data), A->rs, A->cs,
                                                     B->conj, ((T*)B->data), B->rs, B->cs,
                                               beta, C->conj, ((T*)C->data), C->rs, C->cs);
                        }
                    }
                }
            }, comm);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

void tblis_matrix_mult_diag_k(const tblis_comm* comm, const tblis_config* cfg,
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
                                         T(0), ((T*)C->data), C->rs, C->cs);
                    }
                    else if (beta != T(1) || (is_complex<T>::value && C->conj))
                    {
                        internal::scale<T>(comm, get_config(cfg), C->m, C->n,
                                           beta, C->conj,
                                           ((T*)C->data), C->rs, C->cs);
                    }
                }
                else
                {
                    internal::mult<T>(comm, get_config(cfg),
                                      C->m, C->n, A->n,
                                      alpha, A->conj, ((T*)A->data), A->rs, A->cs,
                                             D->conj, ((T*)D->data), D->inc,
                                             B->conj, ((T*)B->data), B->rs, B->cs,
                                       beta, C->conj, ((T*)C->data), C->rs, C->cs);
                }
            }, comm);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

void tblis_matrix_mult_diag_m(const tblis_comm* comm, const tblis_config* cfg,
                              const tblis_vector* D,
                              const tblis_matrix* A,
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
                                      alpha, D->conj, static_cast<const T*>(D->data), D->inc,
                                             A->conj, static_cast<const T*>(A->data), A->rs, A->cs,
                                             B->conj, static_cast<const T*>(B->data), B->rs, B->cs,
                                       beta, C->conj,       static_cast<T*>(C->data), C->rs, C->cs);
                }
            }, comm);
        }

        C->alpha<T>() = T(1);
        C->conj = false;
    })
}

void tblis_matrix_mult_diag_n(const tblis_comm* comm, const tblis_config* cfg,
                              const tblis_matrix* A,
                              const tblis_matrix* B,
                              const tblis_vector* D,
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
                                             B->conj, static_cast<const T*>(B->data), B->rs, B->cs,
                                             D->conj, static_cast<const T*>(D->data), D->inc,
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
