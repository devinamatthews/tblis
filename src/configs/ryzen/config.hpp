#ifndef _TBLIS_CONFIGS_RYZEN_CONFIG_HPP_
#define _TBLIS_CONFIGS_RYZEN_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR( float, bli_sgemm_asm_24x4);
EXTERN_GEMM_UKR( float, bli_sgemm_asm_16x6);
EXTERN_GEMM_UKR( float, bli_sgemm_asm_6x16);
EXTERN_GEMM_UKR( float, bli_sgemm_asm_4x24);

EXTERN_GEMM_UKR(double, bli_dgemm_asm_12x4);
EXTERN_GEMM_UKR(double, bli_dgemm_asm_8x6);
EXTERN_GEMM_UKR(double, bli_dgemm_asm_6x8);
EXTERN_GEMM_UKR(double, bli_dgemm_asm_4x12);

}

namespace tblis
{

extern int ryzen_check();

TBLIS_BEGIN_CONFIG(ryzen)

    TBLIS_CONFIG_GEMM_MR(   6,    6, _, _)
    TBLIS_CONFIG_GEMM_NR(  16,    8, _, _)
    TBLIS_CONFIG_GEMM_KR(   8,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 144,   72, _, _)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, _, _)
    TBLIS_CONFIG_GEMM_KC( 256,  256, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_6x16,
                          bli_dgemm_asm_6x8,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, _, _)

    TBLIS_CONFIG_CHECK(ryzen_check)

TBLIS_END_CONFIG

}

#endif
