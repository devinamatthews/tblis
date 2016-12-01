#ifndef _TBLIS_IFACE_1M_ADD_H_
#define _TBLIS_IFACE_1M_ADD_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_matrix_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_matrix* A, tblis_matrix* B);

#ifdef __cplusplus

}

template <typename T>
void add(T alpha, const_matrix_view<T> A, T beta, matrix_view<T> B)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(beta, B);

    tblis_matrix_add(nullptr, nullptr, &A_s, &B_s);
}

template <typename T>
void add(single_t, T alpha, const_matrix_view<T> A, T beta, matrix_view<T> B)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(beta, B);

    tblis_matrix_add(tblis_single, nullptr, &A_s, &B_s);
}

template <typename T>
void add(const communicator& comm, T alpha, const_matrix_view<T> A,
                                   T  beta,       matrix_view<T> B)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(beta, B);

    tblis_matrix_add(comm, nullptr, &A_s, &B_s);
}

}

#endif

#endif
