#ifndef _TBLIS_REDUCE_HPP_
#define _TBLIS_REDUCE_HPP_

#include "tblis_marray.hpp"

namespace tblis
{

/*******************************************************************************
 *
 * Return the reduction of a tensor, along with the corresponding index (as an
 * offset from A) for MAX, MIN, MAX_ABS, and MIN_ABS reductions
 *
 ******************************************************************************/

template <typename T>
std::pair<T,stride_type> tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A)
{
    std::pair<T,stride_type> p;
    int ret = tensor_reduce(op, A, idx_A, p.first, p.second);
    return p;
}

template <typename T>
T tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, stride_type& idx)
{
    T val;
    int ret = tensor_reduce(op, A, idx_A, val, idx);
    return val;
}

template <typename T>
int tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, T& val)
{
   stride_type idx;
   return tensor_reduce(op, A, idx_A, val, idx);
}

template <typename T>
int tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, T& val, stride_type& idx);

template <typename T>
int tensor_reduce_ref(reduce_t op, const std::vector<len_type>& len_A,
                      const T* A, const std::vector<stride_type>& stride_A,
                      T& val, stride_type& idx);

}

#endif
