#ifndef _TBLIS_IFACE_3T_MULT_H_
#define _TBLIS_IFACE_3T_MULT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT
void tblis_tensor_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_tensor* A, const label_type* idx_A,
                       const tblis_tensor* B, const label_type* idx_B,
                             tblis_tensor* C, const label_type* idx_C);

#if defined(__cplusplus)

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_vector& idx_C)
{
    auto A_(A);
    A_.scalar *= alpha;

    auto C_(C);
    C_.scalar *= beta;

    tblis_tensor_mult(comm, nullptr, &A_, idx_A.data(), &B, idx_B.data(), &C_, idx_C.data());
}

inline
void mult(const communicator& comm,
          const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_vector& idx_C)
{
    mult(comm, {1.0, A.type}, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const tensor& C,
          const label_vector& idx_C)
{
    mult(comm, alpha, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const communicator& comm,
          const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const tensor& C,
          const label_vector& idx_C)
{
    mult(comm, {1.0, A.type}, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor& A,
          const tensor& B,
          const scalar& beta,
          const tensor& C)
{
    label_vector idx_A, idx_B, idx_C;

    TBLIS_ASSERT((A.ndim+B.ndim+C.ndim)%2 == 0);

    auto nAB = (A.ndim+B.ndim-C.ndim)/2;
    auto nAC = (A.ndim+C.ndim-B.ndim)/2;
    auto nBC = (B.ndim+C.ndim-A.ndim)/2;

    for (auto i : range(nAC)) idx_A.push_back(i);
    for (auto i : range(nAC)) idx_C.push_back(i);
    for (auto i : range(nAB)) idx_A.push_back(nAC+i);
    for (auto i : range(nAB)) idx_B.push_back(nAC+i);
    for (auto i : range(nBC)) idx_B.push_back(nAC+nAB+i);
    for (auto i : range(nBC)) idx_C.push_back(nAC+nAB+i);

    mult(comm, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const communicator& comm,
          const tensor& A,
          const tensor& B,
          const scalar& beta,
          const tensor& C)
{
    mult(comm, {1.0, A.type}, A, B, beta, C);
}

inline
void mult(const communicator& comm,
          const scalar& alpha,
          const tensor& A,
          const tensor& B,
          const tensor& C)
{
    mult(comm, alpha, A, B, {0.0, A.type}, C);
}

inline
void mult(const communicator& comm,
          const tensor& A,
          const tensor& B,
          const tensor& C)
{
    mult(comm, {1.0, A.type}, A, B, {0.0, A.type}, C);
}

inline
void mult(const scalar& alpha,
          const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_vector& idx_C)
{
    mult(*(communicator*)nullptr, alpha, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const scalar& beta,
          const tensor& C,
          const label_vector& idx_C)
{
    mult({1.0, A.type}, A, idx_A, B, idx_B, beta, C, idx_C);
}

inline
void mult(const scalar& alpha,
          const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const tensor& C,
          const label_vector& idx_C)
{
    mult(alpha, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const tensor& A,
          const label_vector& idx_A,
          const tensor& B,
          const label_vector& idx_B,
          const tensor& C,
          const label_vector& idx_C)
{
    mult({1.0, A.type}, A, idx_A, B, idx_B, {0.0, A.type}, C, idx_C);
}

inline
void mult(const scalar& alpha,
          const tensor& A,
          const tensor& B,
          const scalar& beta,
          const tensor& C)
{
    mult(*(communicator*)nullptr, alpha, A, B, beta, C);
}

inline
void mult(const tensor& A,
          const tensor& B,
          const scalar& beta,
          const tensor& C)
{
    mult({1.0, A.type}, A, B, beta, C);
}

inline
void mult(const scalar& alpha,
          const tensor& A,
          const tensor& B,
          const tensor& C)
{
    mult(alpha, A, B, {0.0, A.type}, C);
}

inline
void mult(const tensor& A,
          const tensor& B,
          const tensor& C)
{
    mult({1.0, A.type}, A, B, {0.0, A.type}, C);
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void mult(const communicator& comm,
          T alpha, const dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const dpd_marray_view<      T>& C, const label_vector& idx_C);

template <typename T>
void mult(T alpha, const dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const dpd_marray_view<      T>& C, const label_vector& idx_C)
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

template <typename T>
void mult(const communicator& comm,
          T alpha, const indexed_marray_view<const T>& A, const label_vector& idx_A,
                   const indexed_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const indexed_marray_view<      T>& C, const label_vector& idx_C);

template <typename T>
void mult(T alpha, const indexed_marray_view<const T>& A, const label_vector& idx_A,
                   const indexed_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const indexed_marray_view<      T>& C, const label_vector& idx_C)
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

template <typename T>
void mult(const communicator& comm,
          T alpha, const indexed_dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const indexed_dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const indexed_dpd_marray_view<      T>& C, const label_vector& idx_C);

template <typename T>
void mult(T alpha, const indexed_dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const indexed_dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const indexed_dpd_marray_view<      T>& C, const label_vector& idx_C)
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

#endif

}

#endif

#pragma GCC diagnostic pop

#endif
