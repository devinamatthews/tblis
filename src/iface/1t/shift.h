#ifndef _TBLIS_IFACE_1T_SHIFT_H_
#define _TBLIS_IFACE_1T_SHIFT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_shift(const tblis_comm* comm, const tblis_config* cfg,
                        const tblis_scalar* alpha, tblis_tensor* A, const label_type* idx_A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void shift(T alpha, T beta, varray_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A, beta);

    tblis_tensor_shift(nullptr, nullptr, &alpha_s, &A_s, idx_A);
}

template <typename T>
void shift(const communicator& comm, T alpha, T beta, varray_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A, beta);

    tblis_tensor_shift(comm, nullptr, &alpha_s, &A_s, idx_A);
}

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, dpd_varray_view<T> A, const label_type* idx_A);

template <typename T>
void shift(T alpha, T beta, dpd_varray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            shift(comm, alpha, beta, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, indexed_varray_view<T> A, const label_type* idx_A);

template <typename T>
void shift(T alpha, T beta, indexed_varray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            shift(comm, alpha, beta, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, indexed_dpd_varray_view<T> A, const label_type* idx_A);

template <typename T>
void shift(T alpha, T beta, indexed_dpd_varray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            shift(comm, alpha, beta, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef __cplusplus
}
#endif

#endif
