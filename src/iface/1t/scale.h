#ifndef _TBLIS_IFACE_1T_SCALE_H_
#define _TBLIS_IFACE_1T_SCALE_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_tensor* A, const label_type* idx_A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void scale(T alpha, varray_view<T> A, const label_type* idx_A)
{
    tblis_tensor A_s(alpha, A);

    tblis_tensor_scale(nullptr, nullptr, &A_s, idx_A);
}

template <typename T>
void scale(const communicator& comm, T alpha, varray_view<T> A, const label_type* idx_A)
{
    tblis_tensor A_s(alpha, A);

    tblis_tensor_scale(comm, nullptr, &A_s, idx_A);
}

template <typename T>
void scale(const communicator& comm,
           T alpha, dpd_varray_view<T> A, const label_type* idx_A);

template <typename T>
void scale(T alpha, dpd_varray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            scale(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_varray_view<T> A, const label_type* idx_A);

template <typename T>
void scale(T alpha, indexed_varray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            scale(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_dpd_varray_view<T> A, const label_type* idx_A);

template <typename T>
void scale(T alpha, indexed_dpd_varray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            scale(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef __cplusplus
}
#endif

#endif
