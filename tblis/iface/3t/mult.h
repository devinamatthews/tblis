#ifndef TBLIS_IFACE_3T_MULT_H
#define TBLIS_IFACE_3T_MULT_H

#include <tblis/base/types.h>
#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_tensor* A, const label_type* idx_A,
                       const tblis_tensor* B, const label_type* idx_B,
                             tblis_tensor* C, const label_type* idx_C);

#if TBLIS_ENABLE_CXX

void mult(const communicator& comm,
          const scalar& alpha,
          const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_string& idx_C);

inline
void mult(const communicator& comm,
          const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_string& idx_C)
{
    mult(comm, {1.0, A.type}, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const tensor& C,
          const label_string& idx_C)
{
    mult(comm, alpha, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const communicator& comm,
          const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const tensor& C,
          const label_string& idx_C)
{
    mult(comm, {1.0, A.type}, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

void mult(const communicator& comm,
          const scalar& alpha,
          const const_tensor& A,
          const const_tensor& B,
          const scalar& beta,
          const tensor& C);

inline
void mult(const communicator& comm,
          const const_tensor& A,
          const const_tensor& B,
          const scalar& beta,
          const tensor& C)
{
    mult(comm, {1.0, A.type}, A, B, beta, C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const const_tensor& A,
          const const_tensor& B,
          const tensor& C)
{
    mult(comm, alpha, A, B, {0.0, A.type}, C);
}

inline
void mult(const communicator& comm,
          const const_tensor& A,
          const const_tensor& B,
          const tensor& C)
{
    mult(comm, {1.0, A.type}, A, B, {0.0, A.type}, C);
}

inline
void mult(const scalar& alpha,
          const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_string& idx_C)
{
    mult(*(communicator*)nullptr, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_string& idx_C)
{
    mult({1.0, A.type}, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const scalar& alpha,
          const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const tensor& C,
          const label_string& idx_C)
{
    mult(alpha, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const const_tensor& A,
          const label_string& idx_A,
          const const_tensor& B,
          const label_string& idx_B,
          const tensor& C,
          const label_string& idx_C)
{
    mult({1.0, A.type}, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const scalar& alpha,
          const const_tensor& A,
          const const_tensor& B,
          const scalar& beta,
          const tensor& C)
{
    mult(*(communicator*)nullptr, alpha, A, B, beta, C);
}

inline
void mult(const const_tensor& A,
          const const_tensor& B,
          const scalar& beta,
          const tensor& C)
{
    mult({1.0, A.type}, A, B, beta, C);
}

inline
void mult(const scalar& alpha,
          const const_tensor& A,
          const const_tensor& B,
          const tensor& C)
{
    mult(alpha, A, B, {0.0, A.type}, C);
}

inline
void mult(const const_tensor& A,
          const const_tensor& B,
          const tensor& C)
{
    mult({1.0, A.type}, A, B, {0.0, A.type}, C);
}

#ifdef MARRAY_DPD_VARRAY_VIEW_HPP

template <typename T>
void mult(const communicator& comm,
          T alpha, const MArray::dpd_varray_view<const T>& A, const label_string& idx_A,
                   const MArray::dpd_varray_view<const T>& B, const label_string& idx_B,
          T  beta, const MArray::dpd_varray_view<      T>& C, const label_string& idx_C);

template <typename T>
void mult(T alpha, const MArray::dpd_varray_view<const T>& A, const label_string& idx_A,
                   const MArray::dpd_varray_view<const T>& B, const label_string& idx_B,
          T  beta, const MArray::dpd_varray_view<      T>& C, const label_string& idx_C)
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

#endif //MARRAY_DPD_VARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_VARRAY_VIEW_HPP

template <typename T>
void mult(const communicator& comm,
          T alpha, const MArray::indexed_varray_view<const T>& A, const label_string& idx_A,
                   const MArray::indexed_varray_view<const T>& B, const label_string& idx_B,
          T  beta, const MArray::indexed_varray_view<      T>& C, const label_string& idx_C);

template <typename T>
void mult(T alpha, const MArray::indexed_varray_view<const T>& A, const label_string& idx_A,
                   const MArray::indexed_varray_view<const T>& B, const label_string& idx_B,
          T  beta, const MArray::indexed_varray_view<      T>& C, const label_string& idx_C)
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

#endif //MARRAY_INDEXED_VARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP

template <typename T>
void mult(const communicator& comm,
          T alpha, const MArray::indexed_dpd_varray_view<const T>& A, const label_string& idx_A,
                   const MArray::indexed_dpd_varray_view<const T>& B, const label_string& idx_B,
          T  beta, const MArray::indexed_dpd_varray_view<      T>& C, const label_string& idx_C);

template <typename T>
void mult(T alpha, const MArray::indexed_dpd_varray_view<const T>& A, const label_string& idx_A,
                   const MArray::indexed_dpd_varray_view<const T>& B, const label_string& idx_B,
          T  beta, const MArray::indexed_dpd_varray_view<      T>& C, const label_string& idx_C)
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

#endif //MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif //TBLIS_IFACE_3T_MULT_H
