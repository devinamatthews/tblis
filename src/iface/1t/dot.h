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

template <typename T>
void dot(const_tensor_view<T> A, const label_type* idx_A,
         const_tensor_view<T> B, const label_type* idx_B, T& result)
{
    tblis_tensor A_s(A);
    tblis_tensor B_s(B);
    tblis_scalar result_s(result);
    tblis_tensor_dot(nullptr, nullptr, &A_s, idx_A, &B_s, idx_B, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(single_t, const_tensor_view<T> A, const label_type* idx_A,
                     const_tensor_view<T> B, const label_type* idx_B, T& result)
{
    tblis_tensor A_s(A);
    tblis_tensor B_s(B);
    tblis_scalar result_s(result);
    tblis_tensor_dot(tblis_single, nullptr, &A_s, idx_A, &B_s, idx_B, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(const communicator& comm,
         const_tensor_view<T> A, const label_type* idx_A,
         const_tensor_view<T> B, const label_type* idx_B, T& result)
{
    tblis_tensor A_s(A);
    tblis_tensor B_s(B);
    tblis_scalar result_s(result);
    tblis_tensor_dot(comm, nullptr, &A_s, idx_A, &B_s, idx_B, &result_s);
    result = result_s.get<T>();
}

template <typename T>
T dot(const_tensor_view<T> A, const label_type* idx_A,
      const_tensor_view<T> B, const label_type* idx_B)
{
    T result;
    dot(A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(single_t, const_tensor_view<T> A, const label_type* idx_A,
                  const_tensor_view<T> B, const label_type* idx_B)
{
    T result;
    dot(single, A, idx_A, B, idx_B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm,
      const_tensor_view<T> A, const label_type* idx_A,
      const_tensor_view<T> B, const label_type* idx_B)
{
    T result;
    dot(comm, A, idx_A, B, idx_B, result);
    return result;
}

}

#endif

#endif
