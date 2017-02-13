#ifndef _TBLIS_CONFIGS_BULLDOZER_CONFIG_HPP_
#define _TBLIS_CONFIGS_BULLDOZER_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR(          float, bli_sgemm_asm_8x8_fma4);
EXTERN_GEMM_UKR(         double, bli_dgemm_asm_4x6_fma4);
EXTERN_GEMM_UKR(tblis::scomplex, bli_cgemm_asm_8x4_fma4);
EXTERN_GEMM_UKR(tblis::dcomplex, bli_zgemm_asm_4x4_fma4);

}

namespace tblis
{

extern int bulldozer_check();

TBLIS_BEGIN_CONFIG(bulldozer)

TBLIS_CONFIG_GEMM_MR(   8,    4,    8,    4)
TBLIS_CONFIG_GEMM_NR(   8,    6,    4,    4)
TBLIS_CONFIG_GEMM_KR(   8,    4,    4,    4)
TBLIS_CONFIG_GEMM_MC( 128, 1080,   96,   64)
TBLIS_CONFIG_GEMM_NC(4096, 8400, 4096, 4096)
TBLIS_CONFIG_GEMM_KC( 384,  120,  256,  192)

TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_8x8_fma4,
                      bli_dgemm_asm_4x6_fma4,
                      bli_cgemm_asm_8x4_fma4,
                      bli_zgemm_asm_4x4_fma4)

TBLIS_CONFIG_CHECK(bulldozer_check)

TBLIS_END_CONFIG

}

#endif
