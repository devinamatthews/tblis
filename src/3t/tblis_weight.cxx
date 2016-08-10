#include "tblis_weight.hpp"

#include "tblis_config.hpp"
#include "tblis_assert.hpp"
#include "tblis_diagonal.hpp"
#include "tblis_fold.hpp"

namespace tblis
{

template <typename T>
int tensor_weight(T alpha, const_tensor_view<T> A, std::string idx_A,
                           const_tensor_view<T> B, std::string idx_B,
                  T  beta,       tensor_view<T> C, std::string idx_C)
{
    TBLIS_ASSERT(A.dimension() == idx_A.size());
    TBLIS_ASSERT(B.dimension() == idx_B.size());
    TBLIS_ASSERT(C.dimension() == idx_C.size());

    auto len_A = A.lengths();
    auto len_B = B.lengths();
    auto len_C = C.lengths();
    auto stride_A = A.strides();
    auto stride_B = B.strides();
    auto stride_C = C.strides();

    diagonal(len_A, idx_A, stride_A);
    diagonal(len_B, idx_B, stride_B);
    diagonal(len_C, idx_C, stride_C);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto len_ABC = stl_ext::select_from(len_A, idx_A, idx_ABC);
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_B, idx_B, idx_ABC));
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_C, idx_C, idx_ABC));
    auto stride_A_ABC = stl_ext::select_from(stride_A, idx_A, idx_ABC);
    auto stride_B_ABC = stl_ext::select_from(stride_B, idx_B, idx_ABC);
    auto stride_C_ABC = stl_ext::select_from(stride_C, idx_C, idx_ABC);

    TBLIS_ASSERT(stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC).empty());

    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto len_AC = stl_ext::select_from(len_A, idx_A, idx_AC);
    TBLIS_ASSERT(len_AC == stl_ext::select_from(len_C, idx_C, idx_AC));
    auto stride_A_AC = stl_ext::select_from(stride_A, idx_A, idx_AC);
    auto stride_C_AC = stl_ext::select_from(stride_C, idx_C, idx_AC);

    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto len_BC = stl_ext::select_from(len_B, idx_B, idx_BC);
    TBLIS_ASSERT(len_BC == stl_ext::select_from(len_C, idx_C, idx_BC));
    auto stride_B_BC = stl_ext::select_from(stride_B, idx_B, idx_BC);
    auto stride_C_BC = stl_ext::select_from(stride_C, idx_C, idx_BC);

    TBLIS_ASSERT(stl_ext::exclusion(idx_A, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::exclusion(idx_B, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::exclusion(idx_C, idx_AC, idx_BC).empty());

    fold(len_ABC, idx_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    fold(len_AC, idx_AC, stride_A_AC, stride_C_AC);
    fold(len_BC, idx_BC, stride_B_BC, stride_C_BC);

    return impl::tensor_weight_impl(len_AC, len_BC, len_ABC,
                                    alpha, A.data(), stride_A_AC, stride_A_ABC,
                                           B.data(), stride_B_BC, stride_B_ABC,
                                     beta, C.data(), stride_C_AC, stride_C_BC,
                                                     stride_C_ABC);
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_weight(T alpha, const_tensor_view<T> A, std::string idx_A, \
                           const_tensor_view<T> B, std::string idx_B, \
                  T  beta,       tensor_view<T> C, std::string idx_C);
#include "tblis_instantiate_for_types.hpp"

}
