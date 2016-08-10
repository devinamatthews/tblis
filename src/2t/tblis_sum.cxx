#include "tblis_sum.hpp"

#include "tblis_config.hpp"
#include "tblis_assert.hpp"
#include "tblis_diagonal.hpp"
#include "tblis_fold.hpp"

namespace tblis
{

template <typename T>
int tensor_sum(T alpha, const_tensor_view<T> A, std::string idx_A,
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

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto len_A_only = stl_ext::select_from(len_A, idx_A, idx_A_only);
    auto stride_A_only = stl_ext::select_from(stride_A, idx_A, idx_A_only);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);
    auto len_B_only = stl_ext::select_from(len_B, idx_B, idx_B_only);
    auto stride_B_only = stl_ext::select_from(stride_B, idx_B, idx_B_only);

    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);
    fold(len_A_only, idx_A_only, stride_A_only);
    fold(len_B_only, idx_B_only, stride_B_only);

    return tensor_sum_ref(len_A_only, len_B_only, len_AB,
                          alpha, A.data(), stride_A_only, stride_A_AB,
                           beta, B.data(), stride_B_only, stride_B_AB);
}

#define INSTANTIATE_FOR_TYPE(T) \
template <typename T> \
int tensor_sum(T alpha, const_tensor_view<T> A, std::string idx_A, \
               T  beta,       tensor_view<T> B, std::string idx_B);
#include "tblis_instantiate_for_types.hpp"

}
