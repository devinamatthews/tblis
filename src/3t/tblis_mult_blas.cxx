#include "tblis_mult.hpp"

#include "tblis_gemm.h"
#include "tblis_matricize.h"
#include "tblis_replicate.h"
#include "tblis_trace.h"

namespace tblis
{

template <typename T>
int tensor_mult_blas(const std::vector<len_type>& len_A,
                     const std::vector<len_type>& len_B,
                     const std::vector<len_type>& len_C,
                     const std::vector<len_type>& len_AB,
                     const std::vector<len_type>& len_AC,
                     const std::vector<len_type>& len_BC,
                     const std::vector<len_type>& len_ABC,
                     T alpha, const T* A, const std::vector<stride_type>& stride_A_A,
                                          const std::vector<stride_type>& stride_A_AB,
                                          const std::vector<stride_type>& stride_A_AC,
                                          const std::vector<stride_type>& stride_A_ABC,
                              const T* B, const std::vector<stride_type>& stride_B_B,
                                          const std::vector<stride_type>& stride_B_AB,
                                          const std::vector<stride_type>& stride_B_BC,
                                          const std::vector<stride_type>& stride_B_ABC,
                     T  beta,       T* C, const std::vector<stride_type>& stride_C_C,
                                          const std::vector<stride_type>& stride_C_AC,
                                          const std::vector<stride_type>& stride_C_BC,
                                          const std::vector<stride_type>& stride_C_ABC)
{
    tensor<T> ar(len_AC+len_AB);
    tensor<T> br(len_AB+len_BC);
    tensor<T> cr(len_AC+len_BC);

    matrix_view<T> am, bm, cm;

    matricize<T>(ar, am, len_AC.size());
    matricize<T>(br, bm, len_AB.size());
    matricize<T>(cr, cm, len_AC.size());

    MArray::viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (it.next(A, B, C))
    {
        tensor_trace_ref<T>(len_A, ar.lengths(),
                            1.0, A, stride_A_A, stride_A_AC+stride_A_AB,
                            0.0, ar.data(), ar.strides());

        tensor_trace_ref<T>(len_B, br.lengths(),
                            1.0, B, stride_B_B, stride_B_AB+stride_B_BC,
                            0.0, br.data(), br.strides());

        tblis_gemm<T>(alpha, am, bm, 0.0, cm);

        tensor_replicate_ref<T>(len_C, cr.lengths(),
                                 1.0, cr.data(), cr.strides(),
                                beta, C, stride_C_C, stride_C_AC+stride_C_BC);
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_mult_blas(const std::vector<idx_type>& len_A, \
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
