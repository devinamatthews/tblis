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
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void mult(T alpha, varray_view<const T> A, const label_type* idx_A,
                   varray_view<const T> B, const label_type* idx_B,
          T  beta,       varray_view<T> C, const label_type* idx_C)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(B);
    tblis_tensor C_s(beta, C);

    tblis_tensor_mult(nullptr, nullptr, &A_s, idx_A, &B_s, idx_B, &C_s, idx_C);
}

template <typename T>
void mult(const communicator& comm,
          T alpha, varray_view<const T> A, const label_type* idx_A,
                   varray_view<const T> B, const label_type* idx_B,
          T  beta,       varray_view<T> C, const label_type* idx_C)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(B);
    tblis_tensor C_s(beta, C);

    tblis_tensor_mult(comm, nullptr, &A_s, idx_A, &B_s, idx_B, &C_s, idx_C);
}

template <typename T>
void mult(const communicator& comm,
          T alpha, dpd_varray_view<const T> A, const label_type* idx_A,
                   dpd_varray_view<const T> B, const label_type* idx_B,
          T  beta, dpd_varray_view<      T> C, const label_type* idx_C);

template <typename T>
void mult(T alpha, dpd_varray_view<const T> A, const label_type* idx_A,
                   dpd_varray_view<const T> B, const label_type* idx_B,
          T  beta, dpd_varray_view<      T> C, const label_type* idx_C)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void mult(const communicator& comm,
          T alpha, indexed_varray_view<const T> A, const label_type* idx_A,
                   indexed_varray_view<const T> B, const label_type* idx_B,
          T  beta, indexed_varray_view<      T> C, const label_type* idx_C);

template <typename T>
void mult(T alpha, indexed_varray_view<const T> A, const label_type* idx_A,
                   indexed_varray_view<const T> B, const label_type* idx_B,
          T  beta, indexed_varray_view<      T> C, const label_type* idx_C)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void mult(const communicator& comm,
          T alpha, indexed_dpd_varray_view<const T> A, const label_type* idx_A,
                   indexed_dpd_varray_view<const T> B, const label_type* idx_B,
          T  beta, indexed_dpd_varray_view<      T> C, const label_type* idx_C);

template <typename T>
void mult(T alpha, indexed_dpd_varray_view<const T> A, const label_type* idx_A,
                   indexed_dpd_varray_view<const T> B, const label_type* idx_B,
          T  beta, indexed_dpd_varray_view<      T> C, const label_type* idx_C)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef __cplusplus
}
#endif

#endif
