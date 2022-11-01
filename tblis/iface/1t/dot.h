#ifndef TBLIS_IFACE_1T_DOT_H
#define TBLIS_IFACE_1T_DOT_H

#include <tblis/base/types.h>
#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference"

TBLIS_BEGIN_NAMESPACE

TBLIS_EXPORT
void tblis_tensor_dot(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                      const tblis_tensor* B,
                      const label_type* idx_B,
                      tblis_scalar* result);

#if TBLIS_ENABLE_CXX

inline
void dot(const communicator& comm,
         const const_tensor& A,
         const label_string& idx_A,
         const const_tensor& B,
         const label_string& idx_B,
         tblis_scalar& result)
{
    tblis_tensor_dot(comm, nullptr, &A.tensor_, idx_A.idx, &B.tensor_, idx_B.idx, &result);
}

template <typename T>
void dot(const communicator& comm,
         const const_tensor& A,
         const label_string& idx_A,
         const const_tensor& B,
         const label_string& idx_B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(comm, A, idx_A, B, idx_B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const communicator& comm,
                 const const_tensor& A,
                 const label_string& idx_A,
                 const const_tensor& B,
                 const label_string& idx_B)
{
    tblis_scalar result(0.0, A.type);
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const const_tensor& A,
      const label_string& idx_A,
      const const_tensor& B,
      const label_string& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

inline
void dot(const communicator& comm,
         const const_tensor& A,
         const const_tensor& B,
         tblis_scalar& result)
{
    dot(comm, A, idx(A), B, idx(B), result);
}

template <typename T>
void dot(const communicator& comm,
         const const_tensor& A,
         const const_tensor& B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(comm, A, B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const communicator& comm,
                 const const_tensor& A,
                 const const_tensor& B)
{
    tblis_scalar result(0.0, A.type);
    dot(comm, A, B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const const_tensor& A,
      const const_tensor& B)
{
    T result;
    dot(comm, A, B, result);
    return result;
}

inline
void dot(const const_tensor& A,
         const label_string& idx_A,
         const const_tensor& B,
         const label_string& idx_B,
         tblis_scalar& result)
{
    dot(*(communicator*)nullptr, A, idx_A, B, idx_B, result);
}

template <typename T>
void dot(const const_tensor& A,
         const label_string& idx_A,
         const const_tensor& B,
         const label_string& idx_B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(A, idx_A, B, idx_B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const const_tensor& A,
                 const label_string& idx_A,
                 const const_tensor& B,
                 const label_string& idx_B)
{
    tblis_scalar result(0.0, A.type);
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const const_tensor& A,
      const label_string& idx_A,
      const const_tensor& B,
      const label_string& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

inline
void dot(const const_tensor& A,
         const const_tensor& B,
         tblis_scalar& result)
{
    dot(A, idx(A), B, idx(B), result);
}

template <typename T>
void dot(const const_tensor& A,
         const const_tensor& B,
         T& result)
{
    tblis_scalar result_(0.0, A.type);
    dot(A, B, result_);
    result = result_.get<T>();
}

inline
tblis_scalar dot(const const_tensor& A,
                 const const_tensor& B)
{
    tblis_scalar result(0.0, A.type);
    dot(A, B, result);
    return result;
}

template <typename T>
T dot(const const_tensor& A,
      const const_tensor& B)
{
    T result;
    dot(A, B, result);
    return result;
}

#ifdef MARRAY_DPD_MARRAY_VIEW_HPP

template <typename T>
void dot(const communicator& comm,
         MArray::dpd_marray_view<const T> A, const label_string& idx_A,
         MArray::dpd_marray_view<const T> B, const label_string& idx_B, T& result);

template <typename T>
void dot(MArray::dpd_marray_view<const T> A, const label_string& idx_A,
         MArray::dpd_marray_view<const T> B, const label_string& idx_B, T& result)
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
T dot(MArray::dpd_marray_view<const T> A, const label_string& idx_A,
      MArray::dpd_marray_view<const T> B, const label_string& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      MArray::dpd_marray_view<const T> A, const label_string& idx_A,
      MArray::dpd_marray_view<const T> B, const label_string& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif //MARRAY_DPD_MARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_MARRAY_VIEW_HPP

template <typename T>
void dot(const communicator& comm,
         MArray::indexed_marray_view<const T> A, const label_string& idx_A,
         MArray::indexed_marray_view<const T> B, const label_string& idx_B, T& result);

template <typename T>
void dot(MArray::indexed_marray_view<const T> A, const label_string& idx_A,
         MArray::indexed_marray_view<const T> B, const label_string& idx_B, T& result)
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
T dot(MArray::indexed_marray_view<const T> A, const label_string& idx_A,
      MArray::indexed_marray_view<const T> B, const label_string& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      MArray::indexed_marray_view<const T> A, const label_string& idx_A,
      MArray::indexed_marray_view<const T> B, const label_string& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif //MARRAY_INDEXED_MARRAY_VIEW_HPP

#ifdef MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

template <typename T>
void dot(const communicator& comm,
         MArray::indexed_dpd_marray_view<const T> A, const label_string& idx_A,
         MArray::indexed_dpd_marray_view<const T> B, const label_string& idx_B, T& result);

template <typename T>
void dot(MArray::indexed_dpd_marray_view<const T> A, const label_string& idx_A,
         MArray::indexed_dpd_marray_view<const T> B, const label_string& idx_B, T& result)
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
T dot(MArray::indexed_dpd_marray_view<const T> A, const label_string& idx_A,
      MArray::indexed_dpd_marray_view<const T> B, const label_string& idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      MArray::indexed_dpd_marray_view<const T> A, const label_string& idx_A,
      MArray::indexed_dpd_marray_view<const T> B, const label_string& idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif //MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#pragma GCC diagnostic pop

#endif //TBLIS_IFACE_1T_DOT_H
