#include "tblis_contract.hpp"

#include "tblis_config.hpp"
#include "tblis_gemm.hpp"
#include "tblis_matricize.hpp"
#include "tblis_transpose.hpp"

namespace tblis
{

template <typename T>
int tensor_contract_blas(const std::vector<idx_type>& len_AB,
                         const std::vector<idx_type>& len_AC,
                         const std::vector<idx_type>& len_BC,
                         T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                                              const std::vector<stride_type>& stride_A_AC,
                                  const T* B, const std::vector<stride_type>& stride_B_AB,
                                              const std::vector<stride_type>& stride_B_BC,
                         T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                              const std::vector<stride_type>& stride_C_BC)
{
    tensor<T> ar(len_AC+len_AB);
    tensor<T> br(len_AB+len_BC);
    tensor<T> cr(len_AC+len_BC);

    matrix_view<T> am, bm, cm;

    matricize<T>(ar, am, len_AC.size());
    matricize<T>(br, bm, len_AB.size());
    matricize<T>(cr, cm, len_AC.size());

    tensor_transpose_ref<T>(ar.lengths(),
                            1.0, A, stride_A_AC+stride_A_AB,
                            0.0, ar.data(), ar.strides());

    tensor_transpose_ref<T>(br.lengths(),
                            1.0, B, stride_B_AB+stride_B_BC,
                            0.0, br.data(), br.strides());

    tblis_gemm<T>(alpha, am, bm, 0.0, cm);

    tensor_transpose_ref<T>(cr.lengths(),
                             1.0, cr.data(), cr.strides(),
                            beta, C, stride_C_AC, stride_C_BC);

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_contract_blas(const std::vector<idx_type>& len_AB, \
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
}
