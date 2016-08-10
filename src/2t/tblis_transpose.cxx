#include "tblis_transpose.hpp"

#include "tblis_config.hpp"
#include "tblis_assert.hpp"
#include "tblis_diagonal.hpp"
#include "tblis_fold.hpp"

namespace tblis
{

template <typename T>
int tensor_transpose(T alpha, const_tensor_view<T> A, std::string idx_A,
                     T  beta,       tensor_view<T> B, std::string idx_B)
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

    return tensor_transpose_ref(len_A, alpha, A.data(), stride_A,
                                beta, B.data(), stride_B);
}

#define INSTANTIATE_FOR_TYPE(T) \
template <typename T> \
int tensor_transpose(T alpha, const_tensor_view<T> A, std::string idx_A, \
                     T  beta,       tensor_view<T> B, std::string idx_B);
#include "tblis_instantiate_for_types.hpp"

}
