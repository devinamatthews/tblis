#include "tblis_outer_prod.hpp"

#include "tblis_gemm.hpp"
#include "tblis_matricize.hpp"
#include "tblis_transpose.hpp"

namespace tblis
{

template <typename T>
int tensor_outer_prod_blas(const std::vector<len_type>& len_AC,
                           const std::vector<len_type>& len_BC,
                           T alpha, const T* A, const std::vector<stride_type>& stride_A_AC,
                                    const T* B, const std::vector<stride_type>& stride_B_BC,
                           T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                                const std::vector<stride_type>& stride_C_BC)
{
    tensor<T> ar(len_AC);
    tensor<T> br(len_BC);
    tensor<T> cr(len_AC+len_BC);

    matrix_view<T> am, bm, cm;

    matricize<T>(ar, am, len_AC.size());
    matricize<T>(br, bm, 0);
    matricize<T>(cr, cm, len_AC.size());

    tensor_transpose_impl<T>(ar.lengths(),
                             1.0, A, stride_A_AC,
                             0.0, ar.data(), ar.strides());

    tensor_transpose_impl<T>(br.lengths(),
                             1.0, B, stride_B_BC,
                             0.0, br.data(), br.strides());

    tblis_gemm<T>(alpha, am, bm, 0.0, cm);

    tensor_transpose_impl<T>(cr.lengths(),
                              1.0, cr.data(), cr.strides(),
                             beta, C, stride_C_AC+stride_C_BC);

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_outer_prod_blas(const std::vector<idx_type>& len_AC, \
                           const std::vector<idx_type>& len_BC, \
                           T alpha, const T* A, const std::vector<stride_type>& stride_A_AC, \
                                    const T* B, const std::vector<stride_type>& stride_B_BC, \
                           T  beta,       T* C, const std::vector<stride_type>& stride_C_AC, \
                                                const std::vector<stride_type>& stride_C_BC);
#include "tblis_instantiate_for_types.hpp"

}
}
