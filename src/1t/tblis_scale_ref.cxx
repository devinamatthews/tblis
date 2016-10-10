#include "tblis_scale.hpp"

#include "tblis_config.hpp"

namespace tblis
{

template <typename T>
int tensor_scale_ref(const std::vector<len_type>& len_A,
                     T alpha, T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A)
{
    MArray::viterator<> iter_A(len_A, stride_A);

    if (alpha == T(0))
    {
        while (iter_A.next(A))
        {
            *A = T();
        }
    }
    else if (alpha == T(1))
    {
        // do nothing
    }
    else
    {
        while (iter_A.next(A))
        {
            *A *= alpha;
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_scale_ref(const vector<idx_type>& len_A, \
                     T alpha, T* A, const vector<stride_type>& stride_A);
#include "tblis_instantiate_for_types.hpp"

}
