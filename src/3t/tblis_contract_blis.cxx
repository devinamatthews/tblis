#include "../../tblis_config.h"
#include "../tensor/tensor.hpp"
#include "tblis_contract.hpp"

#include "tblis_gemm_template.hpp"
#include "tblis_matrify.hpp"

namespace tblis
{

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
using TensorGEMM =
    typename GEMM<Config,
                  PartitionNC,
                  PartitionKC,
                  MatrifyAndPackB,
                  PartitionMC,
                  MatrifyAndPackA,
                  MatrifyC,
                  PartitionNR,
                  PartitionMR,
                  MicroKernel>::template run<T>;

template <typename T>
int tensor_contract_blis(const std::vector<len_type>& len_AB,
                         const std::vector<len_type>& len_AC,
                         const std::vector<len_type>& len_BC,
                         T alpha, const T* A, const std::vector<stride_type>& stride_A_AB,
                                              const std::vector<stride_type>& stride_A_AC,
                                  const T* B, const std::vector<stride_type>& stride_B_AB,
                                              const std::vector<stride_type>& stride_B_BC,
                         T  beta,       T* C, const std::vector<stride_type>& stride_C_AC,
                                              const std::vector<stride_type>& stride_C_BC)
{
    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_AB, reorder_AB),
                        const_cast<T*>(A),
                        stl_ext::permuted(stride_A_AC, reorder_AC),
                        stl_ext::permuted(stride_A_AB, reorder_AB));

    tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                        stl_ext::permuted(len_BC, reorder_BC),
                        const_cast<T*>(B),
                        stl_ext::permuted(stride_B_AB, reorder_AB),
                        stl_ext::permuted(stride_B_BC, reorder_BC));

    tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_BC, reorder_BC),
                        C,
                        stl_ext::permuted(stride_C_AC, reorder_AC),
                        stl_ext::permuted(stride_C_BC, reorder_BC));

    TensorGEMM<T>()(alpha, at, bt, beta, ct);

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_contract_blis(const std::vector<idx_type>& len_AB, \
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
