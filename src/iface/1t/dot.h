#ifndef _TBLIS_IFACE_1T_DOT_H_
#define _TBLIS_IFACE_1T_DOT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A,
                      const tblis_tensor* B, const label_type* idx_B,
                      tblis_scalar* result);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void dot(varray_view<const T> A, const label_type* idx_A,
         varray_view<const T> B, const label_type* idx_B, T& result)
{
    tblis_tensor A_s(A);
    tblis_tensor B_s(B);
    tblis_scalar result_s(result);
    tblis_tensor_dot(nullptr, nullptr, &A_s, idx_A, &B_s, idx_B, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(const communicator& comm,
         varray_view<const T> A, const label_type* idx_A,
         varray_view<const T> B, const label_type* idx_B, T& result)
{
    tblis_tensor A_s(A);
    tblis_tensor B_s(B);
    tblis_scalar result_s(result);
    tblis_tensor_dot(comm, nullptr, &A_s, idx_A, &B_s, idx_B, &result_s);
    result = result_s.get<T>();
}

template <typename T>
T dot(varray_view<const T> A, const label_type* idx_A,
      varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      varray_view<const T> A, const label_type* idx_A,
      varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
void dot(const communicator& comm,
         dpd_varray_view<const T> A, const label_type* idx_A,
         dpd_varray_view<const T> B, const label_type* idx_B, T& result);

template <typename T>
void dot(dpd_varray_view<const T> A, const label_type* idx_A,
         dpd_varray_view<const T> B, const label_type* idx_B, T& result)
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
T dot(dpd_varray_view<const T> A, const label_type* idx_A,
      dpd_varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      dpd_varray_view<const T> A, const label_type* idx_A,
      dpd_varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
void dot(const communicator& comm,
         indexed_varray_view<const T> A, const label_type* idx_A,
         indexed_varray_view<const T> B, const label_type* idx_B, T& result);

template <typename T>
void dot(indexed_varray_view<const T> A, const label_type* idx_A,
         indexed_varray_view<const T> B, const label_type* idx_B, T& result)
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
T dot(indexed_varray_view<const T> A, const label_type* idx_A,
      indexed_varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      indexed_varray_view<const T> A, const label_type* idx_A,
      indexed_varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
void dot(const communicator& comm,
         indexed_dpd_varray_view<const T> A, const label_type* idx_A,
         indexed_dpd_varray_view<const T> B, const label_type* idx_B, T& result);

template <typename T>
void dot(indexed_dpd_varray_view<const T> A, const label_type* idx_A,
         indexed_dpd_varray_view<const T> B, const label_type* idx_B, T& result)
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
T dot(indexed_dpd_varray_view<const T> A, const label_type* idx_A,
      indexed_dpd_varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      indexed_dpd_varray_view<const T> A, const label_type* idx_A,
      indexed_dpd_varray_view<const T> B, const label_type* idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
