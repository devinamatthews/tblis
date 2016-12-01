#ifndef _TBLIS_IFACE_1T_SET_H_
#define _TBLIS_IFACE_1T_SET_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_tensor* A, const label_type* idx_A);

#ifdef __cplusplus

}

template <typename T>
void set(T alpha, tensor_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A);

    tblis_tensor_set(nullptr, nullptr, &alpha_s, &A_s, idx_A);
}

template <typename T>
void set(single_t, T alpha, tensor_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A);

    tblis_tensor_set(tblis_single, nullptr, &alpha_s, &A_s, idx_A);
}

template <typename T>
void set(const communicator& comm, T alpha, tensor_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A);

    tblis_tensor_set(comm, nullptr, &alpha_s, &A_s, idx_A);
}

}

#endif

#endif
