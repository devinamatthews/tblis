#ifndef _TBLIS_IFACE_1T_ADD_H_
#define _TBLIS_IFACE_1T_ADD_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT
void tblis_tensor_add(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                            tblis_tensor* B,
                      const label_type* idx_B);

#if defined(__cplusplus)

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor& A_,
         const label_vector& idx_A,
         const scalar& beta,
               tensor&& B,
         const label_vector& idx_B)
{
    auto A(A_);
    A.scalar *= alpha.convert(A.type);
    B.scalar *= beta.convert(B.type);
    tblis_tensor_add(comm, nullptr, &A, idx_A.data(), &B, idx_B.data());
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor& A,
         const label_vector& idx_A,
               tensor&& B,
         const label_vector& idx_B)
{
    add(comm, alpha, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const communicator& comm,
         const tensor& A,
         const label_vector& idx_A,
         const scalar& beta,
               tensor&& B,
         const label_vector& idx_B)
{
    add(comm, {1.0, A.type}, A, idx_A, beta, std::move(B), idx_B);
}

inline
void add(const communicator& comm,
         const tensor& A,
         const label_vector& idx_A,
               tensor&& B,
         const label_vector& idx_B)
{
    add(comm, {1.0, A.type}, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor& A,
         const scalar& beta,
               tensor&& B)
{
    add(comm, alpha, A, idx(A), beta, std::move(B), idx(B));
}

inline
void add(const communicator& comm,
         const scalar& alpha,
         const tensor& A,
               tensor&& B)
{
    add(comm, alpha, A, {0.0, A.type}, std::move(B));
}

inline
void add(const communicator& comm,
         const tensor& A,
         const scalar& beta,
               tensor&& B)
{
    add(comm, {1.0, A.type}, A, beta, std::move(B));
}

inline
void add(const communicator& comm,
         const tensor& A,
               tensor&& B)
{
    add(comm, {1.0, A.type}, A, {0.0, A.type}, std::move(B));
}

inline
void add(const scalar& alpha,
         const tensor& A,
         const label_vector& idx_A,
         const scalar& beta,
               tensor&& B,
         const label_vector& idx_B)
{
    add(*(communicator*)nullptr, alpha, A, idx_A, beta, std::move(B), idx_B);
}

inline
void add(const scalar& alpha,
         const tensor& A,
         const label_vector& idx_A,
               tensor&& B,
         const label_vector& idx_B)
{
    add(alpha, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const tensor& A,
         const label_vector& idx_A,
         const scalar& beta,
               tensor&& B,
         const label_vector& idx_B)
{
    add({1.0, A.type}, A, idx_A, beta, std::move(B), idx_B);
}

inline
void add(const tensor& A,
         const label_vector& idx_A,
               tensor&& B,
         const label_vector& idx_B)
{
    add({1.0, A.type}, A, idx_A, {0.0, A.type}, std::move(B), idx_B);
}

inline
void add(const scalar& alpha,
         const tensor& A,
         const scalar& beta,
               tensor&& B)
{
    add(alpha, A, idx(A), beta, std::move(B), idx(B));
}

inline
void add(const scalar& alpha,
         const tensor& A,
               tensor&& B)
{
    add(alpha, A, {0.0, A.type}, std::move(B));
}

inline
void add(const tensor& A,
         const scalar& beta,
               tensor&& B)
{
    add({1.0, A.type}, A, beta, std::move(B));
}

inline
void add(const tensor& A,
               tensor&& B)
{
    add({1.0, A.type}, A, {0.0, A.type}, std::move(B));
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void add(const communicator& comm,
         T alpha, dpd_varray_view<const T> A, const label_vector& idx_A,
         T  beta, dpd_varray_view<      T> B, const label_vector& idx_B);

template <typename T>
void add(T alpha, dpd_varray_view<const T> A, const label_vector& idx_A,
         T  beta, dpd_varray_view<      T> B, const label_vector& idx_B)
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

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_varray_view<const T> A, const label_vector& idx_A,
         T  beta, indexed_varray_view<      T> B, const label_vector& idx_B);

template <typename T>
void add(T alpha, indexed_varray_view<const T> A, const label_vector& idx_A,
         T  beta, indexed_varray_view<      T> B, const label_vector& idx_B)
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

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_dpd_varray_view<const T> A, const label_vector& idx_A,
         T  beta, indexed_dpd_varray_view<      T> B, const label_vector& idx_B);

template <typename T>
void add(T alpha, indexed_dpd_varray_view<const T> A, const label_vector& idx_A,
         T  beta, indexed_dpd_varray_view<      T> B, const label_vector& idx_B)
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

#endif

}

#endif

#pragma GCC diagnostic pop

#endif
