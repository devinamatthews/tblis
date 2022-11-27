#ifndef TBLIS_IFACE_1T_SHIFT_H
#define TBLIS_IFACE_1T_SHIFT_H

#include <tblis/base/types.h>
#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_shift(const tblis_comm* comm,
                        const tblis_config* cfg,
                        const tblis_scalar* alpha,
                              tblis_tensor* A,
                        const label_type* idx_A);

#if TBLIS_ENABLE_CXX

void shift(const communicator& comm,
           const scalar& alpha_,
           const scalar& beta,
           const tensor& A_,
           const label_string& idx_A);

inline
void shift(const communicator& comm,
           const scalar& alpha,
           const tensor& A,
           const label_string& idx_A)
{
    shift(comm, alpha, {1.0, A.type}, std::move(A), idx_A);
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
           const scalar& beta,
           const tensor& A)
{
    shift(comm, alpha, beta, std::move(A), idx(A));
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
           const tensor& A)
{
    shift(comm, alpha, {1.0, A.type}, std::move(A));
}

inline
void shift(const scalar& alpha,
           const scalar& beta,
           const tensor& A,
           const label_string& idx_A)
{
    shift(parallel, alpha, beta, std::move(A), idx_A);
}

inline
void shift(const scalar& alpha,
           const tensor& A,
           const label_string& idx_A)
{
    shift(alpha, {1.0, A.type}, std::move(A), idx_A);
}

inline
void shift(const scalar& alpha,
           const scalar& beta,
           const tensor& A)
{
    shift(alpha, beta, std::move(A), idx(A));
}

inline
void shift(const scalar& alpha,
           const tensor& A)
{
    shift(alpha, {1.0, A.type}, std::move(A));
}

#ifdef MARRAY_DPD_MARRAY_VIEW_HPP

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, MArray::dpd_marray_view<T> A, const label_string& idx_A);

template <typename T>
void shift(T alpha, T beta, MArray::dpd_marray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_DPD_MARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_MARRAY_VIEW_HPP

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, MArray::indexed_marray_view<T> A, const label_string& idx_A);

template <typename T>
void shift(T alpha, T beta, MArray::indexed_marray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_INDEXED_MARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, MArray::indexed_dpd_marray_view<T> A, const label_string& idx_A);

template <typename T>
void shift(T alpha, T beta, MArray::indexed_dpd_marray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif //TBLIS_IFACE_1T_SHIFT_H
