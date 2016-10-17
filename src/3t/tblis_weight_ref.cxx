#include "../../tblis_config.h"
#include "tblis_weight.hpp"


namespace tblis
{

template <typename T>
int tensor_weight_ref(const std::vector<len_type>& len_AC,
                      const std::vector<len_type>& len_BC,
                      const std::vector<len_type>& len_ABC,
                      T alpha, const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A_AC,
                                                          const std::vector<stride_type>& stride_A_ABC,
                               const T* TBLIS_RESTRICT B, const std::vector<stride_type>& stride_B_BC,
                                                          const std::vector<stride_type>& stride_B_ABC,
                      T  beta,       T* TBLIS_RESTRICT C, const std::vector<stride_type>& stride_C_AC,
                                                          const std::vector<stride_type>& stride_C_BC,
                                                          const std::vector<stride_type>& stride_C_ABC)
{
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    MArray::viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (iter_ABC.next(A, B, C))
    {
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
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_weight_ref(const std::vector<idx_type>& len_AC, \
                      const std::vector<idx_type>& len_BC, \
                      const std::vector<idx_type>& len_ABC, \
                      T alpha, const T* A, const std::vector<stride_type>& stride_A_AC, \
                                           const std::vector<stride_type>& stride_A_ABC, \
                               const T* B, const std::vector<stride_type>& stride_B_BC, \
                                           const std::vector<stride_type>& stride_B_ABC, \
                      T  beta,       T* C, const std::vector<stride_type>& stride_C_AC, \
                                           const std::vector<stride_type>& stride_C_BC, \
                                           const std::vector<stride_type>& stride_C_ABC);
#include "tblis_instantiate_for_types.hpp"

}
