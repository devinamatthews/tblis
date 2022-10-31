#ifndef _TBLIS_IFACE_1T_DOT_H_
#define _TBLIS_IFACE_1T_DOT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

#ifdef __cplusplus
namespace tblis
{
#endif

TBLIS_EXPORT
void tblis_tensor_dot(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                      const tblis_tensor* B,
                      const label_type* idx_B,
                      tblis_scalar* result);

#if defined(__cplusplus)

inline
void dot(const communicator& comm,
         const tensor& A,
         const label_vector& idx_A,
         const tensor& B,
         const label_vector& idx_B,
         tblis_scalar& result)
{
    tblis_tensor_dot(comm, nullptr, &A, idx_A.data(), &B, idx_B.data(), &result);
}

template <typename T>
void dot(const communicator& comm,
         const tensor& A,
         const label_vector& idx_A,
         const tensor& B,
         const label_vector& idx_B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(comm, A, idx_A, B, idx_B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const communicator& comm,
                 const tensor& A,
                 const label_vector& idx_A,
                 const tensor& B,
                 const label_vector& idx_B)
{
    tblis_scalar result(0.0, A.type);
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const tensor& A,
      const label_vector& idx_A,
      const tensor& B,
      const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

inline
void dot(const communicator& comm,
         const tensor& A,
         const tensor& B,
         tblis_scalar& result)
{
    dot(comm, A, idx(A), B, idx(B), result);
}

template <typename T>
void dot(const communicator& comm,
         const tensor& A,
         const tensor& B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(comm, A, B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const communicator& comm,
                 const tensor& A,
                 const tensor& B)
{
    tblis_scalar result(0.0, A.type);
    dot(comm, A, B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const tensor& A,
      const tensor& B)
{
    T result;
    dot(comm, A, B, result);
    return result;
}

inline
void dot(const tensor& A,
         const label_vector& idx_A,
         const tensor& B,
         const label_vector& idx_B,
         tblis_scalar& result)
{
    dot(*(communicator*)nullptr, A, idx_A, B, idx_B, result);
}

template <typename T>
void dot(const tensor& A,
         const label_vector& idx_A,
         const tensor& B,
         const label_vector& idx_B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(A, idx_A, B, idx_B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const tensor& A,
                 const label_vector& idx_A,
                 const tensor& B,
                 const label_vector& idx_B)
{
    tblis_scalar result(0.0, A.type);
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const tensor& A,
      const label_vector& idx_A,
      const tensor& B,
      const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

inline
void dot(const tensor& A,
         const tensor& B,
         tblis_scalar& result)
{
    dot(A, idx(A), B, idx(B), result);
}

template <typename T>
void dot(const tensor& A,
         const tensor& B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(A, B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const tensor& A,
                 const tensor& B)
{
    tblis_scalar result(0.0, A.type);
    dot(A, B, result);
    return result;
}

template <typename T>
T dot(const tensor& A,
      const tensor& B)
{
    T result;
    dot(A, B, result);
    return result;
}

#if !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void dot(const communicator& comm,
         dpd_marray_view<const T> A, const label_vector& idx_A,
         dpd_marray_view<const T> B, const label_vector& idx_B, T& result);

template <typename T>
void dot(dpd_marray_view<const T> A, const label_vector& idx_A,
         dpd_marray_view<const T> B, const label_vector& idx_B, T& result)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            dot(comm, A, idx_A, B, idx_B, result);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
T dot(dpd_marray_view<const T> A, const label_vector& idx_A,
      dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      dpd_marray_view<const T> A, const label_vector& idx_A,
      dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
void dot(const communicator& comm,
         indexed_marray_view<const T> A, const label_vector& idx_A,
         indexed_marray_view<const T> B, const label_vector& idx_B, T& result);

template <typename T>
void dot(indexed_marray_view<const T> A, const label_vector& idx_A,
         indexed_marray_view<const T> B, const label_vector& idx_B, T& result)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            dot(comm, A, idx_A, B, idx_B, result);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
T dot(indexed_marray_view<const T> A, const label_vector& idx_A,
      indexed_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      indexed_marray_view<const T> A, const label_vector& idx_A,
      indexed_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
void dot(const communicator& comm,
         indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
         indexed_dpd_marray_view<const T> B, const label_vector& idx_B, T& result);

template <typename T>
void dot(indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
         indexed_dpd_marray_view<const T> B, const label_vector& idx_B, T& result)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            dot(comm, A, idx_A, B, idx_B, result);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
T dot(indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
      indexed_dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      indexed_dpd_marray_view<const T> A, const label_vector& idx_A,
      indexed_dpd_marray_view<const T> B, const label_vector& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif

}

#endif

#pragma GCC diagnostic pop

#endif
