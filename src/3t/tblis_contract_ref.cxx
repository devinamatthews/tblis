#include "tblis.hpp"

#include "tblis_config.hpp"

namespace tblis
{

template <typename T>
int tensor_contract_ref(const std::vector<idx_type>& len_AB,
                        const std::vector<idx_type>& len_AC,
                        const std::vector<idx_type>& len_BC,
                        T alpha, const T* restrict A, const std::vector<stride_type>& stride_A_AB,
                                                      const std::vector<stride_type>& stride_A_AC,
                                 const T* restrict B, const std::vector<stride_type>& stride_B_AB,
                                                      const std::vector<stride_type>& stride_B_BC,
                        T  beta,       T* restrict C, const std::vector<stride_type>& stride_C_AC,
                                                      const std::vector<stride_type>& stride_C_BC)
{
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);

    while (iter_AC.next(A, C))
    {
        while (iter_BC.next(B, C))
        {
            T temp = T();

            while (iter_AB.next(A, B))
            {
                temp += (*A)*(*B);
            }
            temp *= alpha;

            if (beta == T(0))
            {
                *C = temp;
            }
            else if (beta == T(1))
            {
                *C += temp;
            }
            else
            {
                *C = temp + beta*(*C);
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_contract_ref(const std::vector<idx_type>& len_AB, \
                        const std::vector<idx_type>& len_AC, \
                        const std::vector<idx_type>& len_BC, \
                        T alpha, const T* A, const std::vector<stride_type>& stride_A_AB, \
                                             const std::vector<stride_type>& stride_A_AC, \
                                 const T* B, const std::vector<stride_type>& stride_B_AB, \
                                             const std::vector<stride_type>& stride_B_BC, \
                        T  beta,       T* C, const std::vector<stride_type>& stride_C_AC, \
                                             const std::vector<stride_type>& stride_C_BC);
#include "tblis_instantiate_for_types.hpp"

}
