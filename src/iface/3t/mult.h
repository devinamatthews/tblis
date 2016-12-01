#ifndef _TBLIS_IFACE_3T_MULT_H_
#define _TBLIS_IFACE_3T_MULT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_tensor* A, const label_type* idx_A,
                       const tblis_tensor* B, const label_type* idx_B,
                             tblis_tensor* C, const label_type* idx_C);

#ifdef __cplusplus

}

template <typename T>
void mult(T alpha, const_tensor_view<T> A, const label_type* idx_A,
                   const_tensor_view<T> B, const label_type* idx_B,
          T  beta,       tensor_view<T> C, const label_type* idx_C)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(B);
    tblis_tensor C_s(beta, C);

    tblis_tensor_mult(nullptr, nullptr, &A_s, idx_A, &B_s, idx_B, &C_s, idx_C);
}

template <typename T>
void mult(single_t,
          T alpha, const_tensor_view<T> A, const label_type* idx_A,
                   const_tensor_view<T> B, const label_type* idx_B,
          T  beta,       tensor_view<T> C, const label_type* idx_C)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(B);
    tblis_tensor C_s(beta, C);

    tblis_tensor_mult(tblis_single, nullptr, &A_s, idx_A, &B_s, idx_B, &C_s, idx_C);
}

template <typename T>
void mult(const communicator& comm,
          T alpha, const_tensor_view<T> A, const label_type* idx_A,
                   const_tensor_view<T> B, const label_type* idx_B,
          T  beta,       tensor_view<T> C, const label_type* idx_C)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(B);
    tblis_tensor C_s(beta, C);

    tblis_tensor_mult(comm, nullptr, &A_s, idx_A, &B_s, idx_B, &C_s, idx_C);
}

}

#endif

#endif
