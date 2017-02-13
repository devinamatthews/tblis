#ifndef _TBLIS_CONFIGS_CORE2_CONFIG_HPP_
#define _TBLIS_CONFIGS_CORE2_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR(          float, bli_sgemm_asm_8x4);
EXTERN_GEMM_UKR(         double, bli_dgemm_asm_4x4);
EXTERN_GEMM_UKR(tblis::scomplex, bli_cgemm_asm_4x2);
EXTERN_GEMM_UKR(tblis::dcomplex, bli_zgemm_asm_2x2);

}

namespace tblis
{

extern int core2_check();

TBLIS_BEGIN_CONFIG(core2)

TBLIS_CONFIG_GEMM_MR(   8,    4, 4, 2)
TBLIS_CONFIG_GEMM_NR(   4,    4, 2, 2)
TBLIS_CONFIG_GEMM_KR(   4,    2, 2, 2)
TBLIS_CONFIG_GEMM_MC( 768,  384, _, _)
TBLIS_CONFIG_GEMM_NC(4096, 4096, _, _)
TBLIS_CONFIG_GEMM_KC( 384,  384, _, _)

TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_8x4,
                      bli_dgemm_asm_4x4,
                      bli_cgemm_asm_4x2,
                      bli_zgemm_asm_2x2)

TBLIS_CONFIG_CHECK(core2_check)

TBLIS_END_CONFIG

}

#endif
