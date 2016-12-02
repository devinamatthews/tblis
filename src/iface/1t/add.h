#ifndef _TBLIS_IFACE_1T_ADD_H_
#define _TBLIS_IFACE_1T_ADD_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A,
                            tblis_tensor* B, const label_type* idx_B);

#ifdef __cplusplus

}

template <typename T>
void add(T alpha, const_tensor_view<T> A, const label_type* idx_A,
         T  beta,       tensor_view<T> B, const label_type* idx_B)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(beta, B);

    tblis_tensor_add(nullptr, nullptr, &A_s, idx_A, &B_s, idx_B);
}

template <typename T>
void add(single_t, T alpha, const_tensor_view<T> A, const label_type* idx_A,
                   T  beta,       tensor_view<T> B, const label_type* idx_B)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(beta, B);

    tblis_tensor_add(tblis_single, nullptr, &A_s, idx_A, &B_s, idx_B);
}

template <typename T>
void add(const communicator& comm,
         T alpha, const_tensor_view<T> A, const label_type* idx_A,
         T  beta,       tensor_view<T> B, const label_type* idx_B)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(beta, B);

    tblis_tensor_add(comm, nullptr, &A_s, idx_A, &B_s, idx_B);
}

}

#endif

#endif
