#ifndef TBLIS_IFACE_1T_SET_H
#define TBLIS_IFACE_1T_SET_H

#include <tblis/base/types.h>
#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_set(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_scalar* alpha,
                            tblis_tensor* A,
                      const label_type* idx_A);

#if TBLIS_ENABLE_CXX

inline
void set(const communicator& comm,
         const scalar& alpha_,
         const tensor& A,
         const label_string& idx_A)
{
    auto alpha = alpha_.convert(A.type);
    tblis_tensor_set(comm, nullptr, &alpha, const_cast<tensor*>(&A), idx_A.idx);
}

inline
void set(const communicator& comm,
         const scalar& alpha,
         const tensor& A)
{
    set(comm, alpha, std::move(A), idx(A));
}

inline
void set(const scalar& alpha,
         const tensor& A,
         const label_string& idx_A)
{
    set(*(communicator*)nullptr, alpha, std::move(A), idx_A);
}

inline
void set(const scalar& alpha,
         const tensor& A)
{
    set(alpha, std::move(A), idx(A));
}

#ifdef MARRAY_DPD_VARRAY_VIEW_HPP

template <typename T>
void set(const communicator& comm,
         T alpha, MArray::dpd_varray_view<T> A, const label_string& idx_A);

template <typename T>
void set(T alpha, MArray::dpd_varray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_DPD_VARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_VARRAY_VIEW_HPP

template <typename T>
void set(const communicator& comm,
         T alpha, MArray::indexed_varray_view<T> A, const label_string& idx_A);

template <typename T>
void set(T alpha, MArray::indexed_varray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_INDEXED_VARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP

template <typename T>
void set(const communicator& comm,
         T alpha, MArray::indexed_dpd_varray_view<T> A, const label_string& idx_A);

template <typename T>
void set(T alpha, MArray::indexed_dpd_varray_view<T> A, const label_string& idx_A)
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

#endif //MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif //TBLIS_IFACE_1T_SET_H
