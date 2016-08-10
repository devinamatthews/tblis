#include "tblis_reduce.hpp"

#include "tblis_config.hpp"
#include "tblis_assert.hpp"
#include "tblis_diagonal.hpp"
#include "tblis_fold.hpp"

namespace tblis
{

template <typename T>
int tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, T& val, stride_type& idx)
{
    TBLIS_ASSERT(A.dimension() == idx_A.size());

    auto len_A = A.lengths();
    auto stride_A = A.strides();

    diagonal(len_A, idx_A, stride_A);
    fold(len_A, idx_A, stride_A);

    return tensor_reduce_ref(op, len_A, A.data(), stride_A, val, idx);
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, T& val, stride_type& idx);
#include "tblis_instantiate_for_types.hpp"

}
