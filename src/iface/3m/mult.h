#ifndef _TBLIS_IFACE_3M_MULT_H_
#define _TBLIS_IFACE_3M_MULT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_matrix_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_matrix* A, const tblis_matrix* B,
                       tblis_matrix* C);

#ifdef __cplusplus

}

template <typename T>
void mult(T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
          T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    tblis_matrix_mult(nullptr, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void mult(single_t,
          T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
          T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    tblis_matrix_mult(tblis_single, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void mult(const communicator& comm,
          T alpha, const_matrix_view<T> A, const_matrix_view<T> B,
          T beta, matrix_view<T> C)
{
    tblis_matrix A_s(alpha, A);
    tblis_matrix B_s(B);
    tblis_matrix C_s(beta, C);

    tblis_matrix_mult(comm, nullptr, &A_s, &B_s, &C_s);
}

}

#endif

#endif
