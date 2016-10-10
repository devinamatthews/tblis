#include "tblis_outer_prod.hpp"

#include "tblis_config.hpp"

namespace tblis
{

template <typename T>
int tensor_outer_prod_ref(const std::vector<len_type>& len_AC,
                          const std::vector<len_type>& len_BC,
                          T alpha, const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A_AC,
                                   const T* TBLIS_RESTRICT B, const std::vector<stride_type>& stride_B_BC,
                          T  beta,       T* TBLIS_RESTRICT C, const std::vector<stride_type>& stride_C_AC,
                                                              const std::vector<stride_type>& stride_C_BC)
{
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);

    while (iter_AC.next(A, C))
    {
        if (beta == T(0))
        {
            while (iter_BC.next(B, C))
            {
                *C = alpha*(*A)*(*B);
            }
        }
        else
        {
            while (iter_BC.next(B, C))
            {
                *C = alpha*(*A)*(*B) + beta*(*C);
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_outer_prod_ref(const std::vector<idx_type>& len_AC, \
                          const std::vector<idx_type>& len_BC, \
                          T alpha, const T* A, const std::vector<stride_type>& stride_A_AC, \
                                   const T* B, const std::vector<stride_type>& stride_B_BC, \
                          T  beta,       T* C, const std::vector<stride_type>& stride_C_AC, \
                                               const std::vector<stride_type>& stride_C_BC);
#include "tblis_instantiate_for_types.hpp"

}
