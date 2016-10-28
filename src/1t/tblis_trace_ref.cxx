#include "../../tblis_config.h"
#include "tblis_trace.hpp"


namespace tblis
{
namespace impl
{

template <typename T>
int tensor_trace_ref(const std::vector<len_type>& len_A,
                     const std::vector<len_type>& len_AB,
                     T alpha, const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A_A,
                                                         const std::vector<stride_type>& stride_A_AB,
                     T  beta,       T* TBLIS_RESTRICT B, const std::vector<stride_type>& stride_B_AB)
{
    MArray::viterator<1> iter_A(len_A, stride_A_A);
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    while (iter_AB.next(A, B))
    {
        T temp = T();

        if (alpha != T(0))
        {
            while (iter_A.next(A))
            {
                temp += *A;
            }
            temp *= alpha;
        }

        if (beta == T(0))
        {
            *B = temp;
        }
        else if (beta == T(1))
        {
            *B += temp;
        }
        else
        {
            *B = temp + beta*(*B);
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_trace_ref(const std::vector<idx_type>& len_A, \
                     const std::vector<idx_type>& len_AB, \
                     T alpha, const T* A, const std::vector<stride_type>& stride_A_A, \
                                          const std::vector<stride_type>& stride_A_AB, \
                     T  beta,       T* B, const std::vector<stride_type>& stride_B_AB);
#include "tblis_instantiate_for_types.hpp"

}
}
