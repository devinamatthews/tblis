#ifndef _TBLIS_NODES_GEMM_HPP_
#define _TBLIS_NODES_GEMM_HPP_

#include "partm.hpp"
#include "packm.hpp"
#include "matrify.hpp"
#include "gemm_mkr.hpp"
#include "gemm_ukr.hpp"

namespace tblis
{

extern MemoryPool BuffersForA, BuffersForB, BuffersForScatter;

using GotoGEMM = partition_gemm_nc<
                   partition_gemm_kc<
                     pack_b<BuffersForB,
                       partition_gemm_mc<
                         pack_a<BuffersForA,
                           partition_gemm_nr<
                             partition_gemm_mr<
                               gemm_micro_kernel>>>>>>>;

using GotoGEMM2 = partition_gemm_nc<
                    partition_gemm_kc<
                      pack_b<BuffersForB,
                        partition_gemm_mc<
                          pack_a<BuffersForA,
                           gemm_macro_kernel>>>>>;

using TensorGEMM = partition_gemm_nc<
                     partition_gemm_kc<
                       matrify_and_pack_b<BuffersForB,
                         partition_gemm_mc<
                           matrify_and_pack_a<BuffersForA,
                             matrify_c<BuffersForScatter,
                               partition_gemm_nr<
                                 partition_gemm_mr<
                                   gemm_micro_kernel>>>>>>>>;

}

#endif
