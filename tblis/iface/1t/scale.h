#ifndef TBLIS_IFACE_1T_SCALE_H
#define TBLIS_IFACE_1T_SCALE_H

#include <tblis/base/types.h>
#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT

void tblis_tensor_scale(const tblis_comm* comm,
                        const tblis_config* cfg,
                              tblis_tensor* A,
                        const label_type* idx_A);

#if TBLIS_ENABLE_CXX

void scale(const communicator& comm,
           const scalar& alpha,
           const tensor& A_,
           const label_string& idx_A);

inline
void scale(const communicator& comm,
           const tensor& A,
           const label_string& idx_A)
{
    scale(comm, {1.0, A.type}, A, idx_A);
}

inline
void scale(const communicator& comm,
           const scalar& alpha,
           const tensor& A)
{
    scale(comm, alpha, A, idx(A));
}

inline
void scale(const communicator& comm,
           const tensor& A)
{
    scale(comm, {1.0, A.type}, A);
}

inline
void scale(const scalar& alpha,
           const tensor& A,
           const label_string& idx_A)
{
    scale(parallel, alpha, A, idx_A);
}

inline
void scale(const tensor& A,
           const label_string& idx_A)
{
    scale({1.0, A.type}, A, idx_A);
}

inline
void scale(const scalar& alpha,
           const tensor& A)
{
    scale(alpha, A, idx(A));
}

inline
void scale(const tensor& A)
{
    scale({1.0, A.type}, A);
}

#ifdef MARRAY_DPD_MARRAY_VIEW_HPP

template <typename T>
void scale(const communicator& comm,
           T alpha, MArray::dpd_marray_view<T> A, const label_string& idx_A);

template <typename T>
void scale(T alpha, MArray::dpd_marray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_DPD_MARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_MARRAY_VIEW_HPP

template <typename T>
void scale(const communicator& comm,
           T alpha, MArray::indexed_marray_view<T> A, const label_string& idx_A);

template <typename T>
void scale(T alpha, MArray::indexed_marray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_INDEXED_MARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

template <typename T>
void scale(const communicator& comm,
           T alpha, MArray::indexed_dpd_marray_view<T> A, const label_string& idx_A);

template <typename T>
void scale(T alpha, MArray::indexed_dpd_marray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif //TBLIS_IFACE_1T_SCALE_H
