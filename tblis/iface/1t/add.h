#ifndef TBLIS_IFACE_1T_ADD_H
#define TBLIS_IFACE_1T_ADD_H

#include <tblis/base/types.h>
#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_add(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                            tblis_tensor* B,
                      const label_type* idx_B);

#if TBLIS_ENABLE_CXX

void add(const communicator& comm,
         const scalar& alpha,
         const const_tensor& A_,
         const label_string& idx_A,
         const scalar& beta,
         const tensor& B_,
         const label_string& idx_B);

inline
void add(const communicator& comm,
         const scalar& alpha,
         const const_tensor& A,
         const label_string& idx_A,
         const tensor& B,
         const label_string& idx_B)
{
    add(comm, alpha, A, idx_A, {0.0, A.type}, B, idx_B);
}

inline
void add(const communicator& comm,
         const const_tensor& A,
         const label_string& idx_A,
         const scalar& beta,
         const tensor& B,
         const label_string& idx_B)
{
    add(comm, {1.0, A.type}, A, idx_A, beta, B, idx_B);
}

inline
void add(const communicator& comm,
         const const_tensor& A,
         const label_string& idx_A,
         const tensor& B,
         const label_string& idx_B)
{
    add(comm, {1.0, A.type}, A, idx_A, {0.0, A.type}, B, idx_B);
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const const_tensor& A,
         const scalar& beta,
         const tensor& B)
{
    add(comm, alpha, A, idx(A), beta, B, idx(B));
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const const_tensor& A,
         const tensor& B)
{
    add(comm, alpha, A, {0.0, A.type}, B);
}

inline
void add(const communicator& comm,
         const const_tensor& A,
         const scalar& beta,
         const tensor& B)
{
    add(comm, {1.0, A.type}, A, beta, B);
}

inline
void add(const communicator& comm,
         const const_tensor& A,
         const tensor& B)
{
    add(comm, {1.0, A.type}, A, {0.0, A.type}, B);
}

inline
void add(const scalar& alpha,
         const const_tensor& A,
         const label_string& idx_A,
         const scalar& beta,
         const tensor& B,
         const label_string& idx_B)
{
    add(*(communicator*)nullptr, alpha, A, idx_A, beta, B, idx_B);
}

inline
void add(const scalar& alpha,
         const const_tensor& A,
         const label_string& idx_A,
         const tensor& B,
         const label_string& idx_B)
{
    add(alpha, A, idx_A, {0.0, A.type}, B, idx_B);
}

inline
void add(const const_tensor& A,
         const label_string& idx_A,
         const scalar& beta,
         const tensor& B,
         const label_string& idx_B)
{
    add({1.0, A.type}, A, idx_A, beta, B, idx_B);
}

inline
void add(const const_tensor& A,
         const label_string& idx_A,
         const tensor& B,
         const label_string& idx_B)
{
    add({1.0, A.type}, A, idx_A, {0.0, A.type}, B, idx_B);
}

inline
void add(const scalar& alpha,
         const const_tensor& A,
         const scalar& beta,
         const tensor& B)
{
    add(alpha, A, idx(A), beta, B, idx(B));
}

inline
void add(const scalar& alpha,
         const const_tensor& A,
         const tensor& B)
{
    add(alpha, A, {0.0, A.type}, B);
}

inline
void add(const const_tensor& A,
         const scalar& beta,
         const tensor& B)
{
    add({1.0, A.type}, A, beta, B);
}

inline
void add(const const_tensor& A,
         const tensor& B)
{
    add({1.0, A.type}, A, {0.0, A.type}, B);
}

#ifdef MARRAY_DPD_VARRAY_VIEW_HPP

template <typename T>
void add(const communicator& comm,
         T alpha, MArray::dpd_varray_view<const T> A, const label_string& idx_A,
         T  beta, MArray::dpd_varray_view<      T> B, const label_string& idx_B);

template <typename T>
void add(T alpha, MArray::dpd_varray_view<const T> A, const label_string& idx_A,
         T  beta, MArray::dpd_varray_view<      T> B, const label_string& idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif //MARRAY_DPD_VARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_VARRAY_VIEW_HPP

template <typename T>
void add(const communicator& comm,
         T alpha, MArray::indexed_varray_view<const T> A, const label_string& idx_A,
         T  beta, MArray::indexed_varray_view<      T> B, const label_string& idx_B);

template <typename T>
void add(T alpha, MArray::indexed_varray_view<const T> A, const label_string& idx_A,
         T  beta, MArray::indexed_varray_view<      T> B, const label_string& idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif //MARRAY_INDEXED_VARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP

template <typename T>
void add(const communicator& comm,
         T alpha, MArray::indexed_dpd_varray_view<const T> A, const label_string& idx_A,
         T  beta, MArray::indexed_dpd_varray_view<      T> B, const label_string& idx_B);

template <typename T>
void add(T alpha, MArray::indexed_dpd_varray_view<const T> A, const label_string& idx_A,
         T  beta, MArray::indexed_dpd_varray_view<      T> B, const label_string& idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif //MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif //TBLIS_IFACE_1T_ADD_H
