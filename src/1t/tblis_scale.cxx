#include "tblis_scale.hpp"

#include "../../tblis_config.h"
#include "../util/assert.h"
#include "tblis_diagonal.hpp"
#include "tblis_fold.hpp"

namespace tblis
{

template <typename T>
int tensor_scale(T alpha, tensor_view<T> A, std::string idx_A)
{
    TBLIS_ASSERT(A.dimension() == idx_A.size());

    auto len_A = A.lengths();
    auto stride_A = A.strides();

    diagonal(len_A, idx_A, stride_A);
    fold(len_A, idx_A, stride_A);

    return tensor_scale_ref(len_A, alpha, A.data(), stride_A);
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_scale(T alpha, tensor_view<T> A, std::string idx_A);
#include "tblis_instantiate_for_types.hpp"

}
