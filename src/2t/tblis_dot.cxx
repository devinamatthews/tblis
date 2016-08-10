#include "tblis_dot.hpp"

#include "tblis_config.hpp"
#include "tblis_assert.hpp"
#include "tblis_diagonal.hpp"
#include "tblis_fold.hpp"

namespace tblis
{

template <typename T>
int tensor_dot(const_tensor_view<T> A, std::string idx_A,
               const_tensor_view<T> B, std::string idx_B, T& val)
{
    TBLIS_ASSERT(A.dimension() == idx_A.size());
    TBLIS_ASSERT(B.dimension() == idx_B.size());

    auto len_A = A.lengths();
    auto len_B = B.lengths();
    auto stride_A = A.strides();
    auto stride_B = B.strides();

    diagonal(len_A, idx_A, stride_A);
    diagonal(len_B, idx_B, stride_B);
    TBLIS_ASSERT(idx_A == idx_B);

    fold(len_A, idx_A, stride_A, stride_B);

    return tensor_dot_ref(len_A, A.data(), stride_A,
                                 B.data(), stride_B, val);
}

#define INSTANTIATE_FOR_TYPE(T) \
template <typename T> \
int tensor_dot(const_tensor_view<T> A, std::string idx_A, \
               const_tensor_view<T> B, std::string idx_B, T& val);
#include "tblis_instantiate_for_types.hpp"

}
