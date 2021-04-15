#ifndef _TBLIS_CONFIGS_CORE2_CONFIG_HPP_
#define _TBLIS_CONFIGS_CORE2_CONFIG_HPP_

#include <tblis/internal/configs.hpp>

EXTERN_BLIS_GEMM_UKR(bli_sgemm_asm_8x4);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_asm_4x4);

namespace tblis
{

extern int core2_check();

TBLIS_BEGIN_CONFIG(core2)

TBLIS_CONFIG_GEMM_MR(   8,    4, _, _)
TBLIS_CONFIG_GEMM_NR(   4,    4, _, _)
TBLIS_CONFIG_GEMM_KR(   4,    2, _, _)
TBLIS_CONFIG_GEMM_MC( 768,  384, _, _)
TBLIS_CONFIG_GEMM_NC(4096, 4096, _, _)
TBLIS_CONFIG_GEMM_KC( 384,  384, _, _)

TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_asm_8x4,
                           bli_dgemm_asm_4x4,
                                           _,
                                           _)

TBLIS_CONFIG_CHECK(core2_check)

TBLIS_END_CONFIG

}

#endif
