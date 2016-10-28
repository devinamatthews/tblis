#include "../../tblis_config.h"
#include "dot.h"


namespace tblis
{

template <typename T>
int tensor_dot_ref(const std::vector<len_type>& len_AB,
                   const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A_AB,
                   const T* TBLIS_RESTRICT B, const std::vector<stride_type>& stride_B_AB,
                   T& val)
{
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    val = T(0);

    while (iter_AB.next(A, B))
    {
        val += (*A)*(*B);
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_dot_ref(const std::vector<idx_type>& len_AB, \
                   const T* A, const std::vector<stride_type>& stride_A_AB, \
                   const T* B, const std::vector<stride_type>& stride_B_AB, \
                   T& val);
#include "tblis_instantiate_for_types.hpp"

}
