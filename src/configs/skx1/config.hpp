#ifndef _TBLIS_CONFIGS_SKX1_CONFIG_HPP_
#define _TBLIS_CONFIGS_SKX1_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR( float, bli_sgemm_asm_6x16);
EXTERN_GEMM_UKR(double, bli_dgemm_asm_6x8);

}

namespace tblis
{

extern int skx1_check();

TBLIS_BEGIN_CONFIG(skx1)

    TBLIS_CONFIG_GEMM_MR(   6,    6, _, _)
    TBLIS_CONFIG_GEMM_NR(  16,    8, _, _)
    TBLIS_CONFIG_GEMM_KR(   8,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 288,  144, _, _)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, _, _)
    TBLIS_CONFIG_GEMM_KC( 256,  256, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_6x16,
                          bli_dgemm_asm_6x8,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, _, _)

    TBLIS_CONFIG_CHECK(skx1_check)

TBLIS_END_CONFIG

}

#endif
