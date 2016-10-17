#include "../../tblis_config.h"
#include "tblis_mult.hpp"


namespace tblis
{

template <typename T>
int tensor_mult_ref(const std::vector<len_type>& len_A,
                    const std::vector<len_type>& len_B,
                    const std::vector<len_type>& len_C,
                    const std::vector<len_type>& len_AB,
                    const std::vector<len_type>& len_AC,
                    const std::vector<len_type>& len_BC,
                    const std::vector<len_type>& len_ABC,
                    T alpha, const T* TBLIS_RESTRICT A, const std::vector<stride_type>& stride_A_A,
                                                        const std::vector<stride_type>& stride_A_AB,
                                                        const std::vector<stride_type>& stride_A_AC,
                                                        const std::vector<stride_type>& stride_A_ABC,
                             const T* TBLIS_RESTRICT B, const std::vector<stride_type>& stride_B_B,
                                                        const std::vector<stride_type>& stride_B_AB,
                                                        const std::vector<stride_type>& stride_B_BC,
                                                        const std::vector<stride_type>& stride_B_ABC,
                    T  beta,       T* TBLIS_RESTRICT C, const std::vector<stride_type>& stride_C_C,
                                                        const std::vector<stride_type>& stride_C_AC,
                                                        const std::vector<stride_type>& stride_C_BC,
                                                        const std::vector<stride_type>& stride_C_ABC)
{
    MArray::viterator<1> iter_A(len_A, stride_A_A);
    MArray::viterator<1> iter_B(len_B, stride_B_B);
    MArray::viterator<1> iter_C(len_C, stride_C_C);
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
    MArray::viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
    MArray::viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
    MArray::viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (iter_ABC.next(A, B, C))
    {
        while (iter_AC.next(A, C))
        {
            while (iter_BC.next(B, C))
            {
                T temp = T();

                while (iter_AB.next(A, B))
                {
                    T temp_A = T();
                    while (iter_A.next(A))
                    {
                        temp_A += *A;
                    }

                    T temp_B = T();
                    while (iter_B.next(B))
                    {
                        temp_B += *B;
                    }

                    temp += temp_A*temp_B;
                }

                temp *= alpha;

                if (beta == T(0))
                {
                    while (iter_C.next(C))
                    {
                        *C = temp;
                    }
                }
                else if (beta == T(1))
                {
                    while (iter_C.next(C))
                    {
                        *C += temp;
                    }
                }
                else
                {
                    while (iter_C.next(C))
                    {
                        *C = temp + beta*(*C);
                    }
                }
            }
        }
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_mult_ref(const std::vector<idx_type>& len_A, \
                    const std::vector<idx_type>& len_B, \
                    const std::vector<idx_type>& len_C, \
                    const std::vector<idx_type>& len_AB, \
                    const std::vector<idx_type>& len_AC, \
                    const std::vector<idx_type>& len_BC, \
                    const std::vector<idx_type>& len_ABC, \
                    T alpha, const T* A, const std::vector<stride_type>& stride_A_A, \
                                         const std::vector<stride_type>& stride_A_AB, \
                                         const std::vector<stride_type>& stride_A_AC, \
                                         const std::vector<stride_type>& stride_A_ABC, \
                             const T* B, const std::vector<stride_type>& stride_B_B, \
                                         const std::vector<stride_type>& stride_B_AB, \
                                         const std::vector<stride_type>& stride_B_BC, \
                                         const std::vector<stride_type>& stride_B_ABC, \
                    T  beta,       T* C, const std::vector<stride_type>& stride_C_C, \
                                         const std::vector<stride_type>& stride_C_AC, \
                                         const std::vector<stride_type>& stride_C_BC, \
                                         const std::vector<stride_type>& stride_C_ABC);
#include "tblis_instantiate_for_types.hpp"

}
