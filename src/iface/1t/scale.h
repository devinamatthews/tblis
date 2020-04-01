#ifndef _TBLIS_IFACE_1T_SCALE_H_
#define _TBLIS_IFACE_1T_SCALE_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT

void tblis_tensor_scale(const tblis_comm* comm,
                        const tblis_config* cfg,
                              tblis_tensor* A,
                        const label_type* idx_A);

#if defined(__cplusplus)

inline
void scale(const communicator& comm,
           const scalar& alpha,
                 tensor&& A,
           const label_vector& idx_A)
{
    A.scalar *= alpha.convert(A.type);
    tblis_tensor_scale(comm, nullptr, &A, idx_A.data());
}

inline
void scale(const communicator& comm,
                 tensor&& A,
           const label_vector& idx_A)
{
    scale(comm, {1.0, A.type}, std::move(A), idx_A);
}

inline
void scale(const communicator& comm,
           const scalar& alpha,
                 tensor&& A)
{
    scale(comm, alpha, std::move(A), idx(A));
}

inline
void scale(const communicator& comm,
                 tensor&& A)
{
    scale(comm, {1.0, A.type}, std::move(A));
}

inline
void scale(const scalar& alpha,
                 tensor&& A,
           const label_vector& idx_A)
{
    scale(*(communicator*)nullptr, alpha, std::move(A), idx_A);
}

inline
void scale(      tensor&& A,
           const label_vector& idx_A)
{
    scale({1.0, A.type}, std::move(A), idx_A);
}

inline
void scale(const scalar& alpha,
                 tensor&& A)
{
    scale(alpha, std::move(A), idx(A));
}

inline
void scale(      tensor&& A)
{
    scale({1.0, A.type}, std::move(A));
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void scale(const communicator& comm,
           T alpha, dpd_varray_view<T> A, const label_vector& idx_A);

template <typename T>
void scale(T alpha, dpd_varray_view<T> A, const label_vector& idx_A)
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
           T alpha, indexed_varray_view<T> A, const label_vector& idx_A);

template <typename T>
void scale(T alpha, indexed_varray_view<T> A, const label_vector& idx_A)
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
           T alpha, indexed_dpd_varray_view<T> A, const label_vector& idx_A);

template <typename T>
void scale(T alpha, indexed_dpd_varray_view<T> A, const label_vector& idx_A)
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

}

#endif

#pragma GCC diagnostic pop

#endif
