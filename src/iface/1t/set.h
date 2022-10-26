#ifndef _TBLIS_IFACE_1T_SET_H_
#define _TBLIS_IFACE_1T_SET_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT
void tblis_tensor_set(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_scalar* alpha,
                            tblis_tensor* A,
                      const label_type* idx_A);

#if defined(__cplusplus)

inline
void set(const communicator& comm,
         const scalar& alpha_,
               tensor&& A,
         const label_vector& idx_A)
{
    auto alpha = alpha_.convert(A.type);
    tblis_tensor_set(comm, nullptr, &alpha, &A, idx_A.data());
}

inline
void set(const communicator& comm,
         const scalar& alpha,
               tensor&& A)
{
    set(comm, alpha, std::move(A), idx(A));
}

inline
void set(const scalar& alpha,
               tensor&& A,
         const label_vector& idx_A)
{
    set(*(communicator*)nullptr, alpha, std::move(A), idx_A);
}

inline
void set(const scalar& alpha,
               tensor&& A)
{
    set(alpha, std::move(A), idx(A));
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void set(const communicator& comm,
         T alpha, dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void set(T alpha, dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void set(const communicator& comm,
         T alpha, indexed_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void set(T alpha, indexed_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void set(const communicator& comm,
         T alpha, indexed_dpd_marray_view<T> A, const label_vector& idx_A);

template <typename T>
void set(T alpha, indexed_dpd_marray_view<T> A, const label_vector& idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

}

#endif

#pragma GCC diagnostic pop

#endif
