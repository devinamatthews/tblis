#ifndef _TBLIS_CONFIGS_SANDYBRIDGE_CONFIG_HPP_
#define _TBLIS_CONFIGS_SANDYBRIDGE_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR(          float, bli_sgemm_asm_8x8);
EXTERN_GEMM_UKR(         double, bli_dgemm_asm_8x4);
EXTERN_GEMM_UKR(tblis::scomplex, bli_cgemm_asm_8x4);
EXTERN_GEMM_UKR(tblis::dcomplex, bli_zgemm_asm_4x4);

}

namespace tblis
{

extern int sandybridge_check();

TBLIS_BEGIN_CONFIG(sandybridge)

TBLIS_CONFIG_GEMM_MR(   8,    8,    8,    4)
TBLIS_CONFIG_GEMM_NR(   8,    4,    4,    4)
TBLIS_CONFIG_GEMM_MC( 128,   96,   96,   64)
TBLIS_CONFIG_GEMM_NC(4096, 4096, 4096, 4096)
TBLIS_CONFIG_GEMM_KC( 384,  256,  256,  192)

TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_8x8,
                      bli_dgemm_asm_8x4,
                      bli_cgemm_asm_8x4,
                      bli_zgemm_asm_4x4)

TBLIS_CONFIG_CHECK(sandybridge_check)

TBLIS_END_CONFIG

}

#endif
