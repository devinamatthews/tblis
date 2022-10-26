#ifndef _TBLIS_IFACE_1T_SHIFT_H_
#define _TBLIS_IFACE_1T_SHIFT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT
void tblis_tensor_shift(const tblis_comm* comm,
                        const tblis_config* cfg,
                        const tblis_scalar* alpha,
                              tblis_tensor* A,
                        const label_type* idx_A);

#if defined(__cplusplus)

inline
void shift(const communicator& comm,
           const scalar& alpha_,
           const scalar& beta,
                 tensor&& A,
           const label_vector& idx_A)
{
    auto alpha = alpha_.convert(A.type);
    A.scalar *= beta.convert(A.type);
    tblis_tensor_shift(comm, nullptr, &alpha, &A, idx_A.data());
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
                 tensor&& A,
           const label_vector& idx_A)
{
    shift(comm, alpha, {1.0, A.type}, std::move(A), idx_A);
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
           const scalar& beta,
                 tensor&& A)
{
    shift(comm, alpha, beta, std::move(A), idx(A));
}

inline
void shift(const communicator& comm,
           const scalar& alpha,
                 tensor&& A)
{
    shift(comm, alpha, {1.0, A.type}, std::move(A));
}

inline
void shift(const scalar& alpha,
           const scalar& beta,
                 tensor&& A,
           const label_vector& idx_A)
{
    shift(*(communicator*)nullptr, alpha, beta, std::move(A), idx_A);
}

inline
void shift(const scalar& alpha,
                 tensor&& A,
           const label_vector& idx_A)
{
    shift(alpha, {1.0, A.type}, std::move(A), idx_A);
}

inline
void shift(const scalar& alpha,
           const scalar& beta,
                 tensor&& A)
{
    shift(alpha, beta, std::move(A), idx(A));
}

inline
void shift(const scalar& alpha,
                 tensor&& A)
{
    shift(alpha, {1.0, A.type}, std::move(A));
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void shift(const communicator& comm,
         T alpha, T beta, dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void shift(T alpha, T beta, dpd_marray_view<T> A, const label_vector& idx_A)
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
         T alpha, T beta, indexed_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void shift(T alpha, T beta, indexed_marray_view<T> A, const label_vector& idx_A)
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
         T alpha, T beta, indexed_dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void shift(T alpha, T beta, indexed_dpd_marray_view<T> A, const label_vector& idx_A)
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

}

#endif

#pragma GCC diagnostic pop

#endif
