#ifndef _TBLIS_IFACE_1T_REDUCE_H_
#define _TBLIS_IFACE_1T_REDUCE_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_tensor* A, const label_type* idx_A,
                         tblis_scalar* result, len_type* idx);

#ifdef __cplusplus

}

template <typename T>
void reduce(reduce_t op, const_tensor_view<T> A, const label_type* idx_A, T& result, len_type& idx)
{
    tblis_tensor A_s(A);
    tblis_scalar result_s(result);
    tblis_tensor_reduce(nullptr, nullptr, op, &A_s, idx_A, &result_s, &idx);
    result = result_s.get<T>();
}

template <typename T>
void reduce(single_t, reduce_t op, const_tensor_view<T> A, const label_type* idx_A,
            T& result, len_type& idx)
{
    tblis_tensor A_s(A);
    tblis_scalar result_s(result);
    tblis_tensor_reduce(tblis_single, nullptr, op, &A_s, idx_A, &result_s, &idx);
    result = result_s.get<T>();
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, const_tensor_view<T> A, const label_type* idx_A,
            T& result, len_type& idx)
{
    tblis_tensor A_s(A);
    tblis_scalar result_s(result);
    tblis_tensor_reduce(comm, nullptr, op, &A_s, idx_A, &result_s, &idx);
    result = result_s.get<T>();
}

template <typename T>
std::pair<T,len_type> reduce(reduce_t op, const_tensor_view<T> A, const label_type* idx_A)
{
    std::pair<T,len_type> result;
    reduce(op, A, idx_A, result.first, result.second);
    return result;
}

template <typename T>
std::pair<T,len_type> reduce(single_t, reduce_t op, const_tensor_view<T> A, const label_type* idx_A)
{
    std::pair<T,len_type> result;
    reduce(single, op, A, idx_A, result.first, result.second);
    return result;
}

template <typename T>
std::pair<T,len_type> reduce(const communicator& comm, reduce_t op,
                             const_tensor_view<T> A, const label_type* idx_A)
{
    std::pair<T,len_type> result;
    reduce(comm, op, A, idx_A, result.first, result.second);
    return result;
}

}

#endif

#endif
