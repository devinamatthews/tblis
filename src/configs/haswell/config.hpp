#ifndef _TBLIS_CONFIGS_HASWELL_CONFIG_HPP_
#define _TBLIS_CONFIGS_HASWELL_CONFIG_HPP_

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

extern int haswell_check();

TBLIS_BEGIN_CONFIG(haswell_d12x4)

    TBLIS_CONFIG_GEMM_MR(  24,   12, _, _)
    TBLIS_CONFIG_GEMM_NR(   4,    4, _, _)
    TBLIS_CONFIG_GEMM_KR(   8,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 264,   96, _, _)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, _, _)
    TBLIS_CONFIG_GEMM_KC( 128,  192, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_24x4,
                          bli_dgemm_asm_12x4,
                          _,
                          _)

    TBLIS_CONFIG_CHECK(haswell_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(haswell_d4x12)

    TBLIS_CONFIG_GEMM_MR(   4,    4, _, _)
    TBLIS_CONFIG_GEMM_NR(  24,   12, _, _)
    TBLIS_CONFIG_GEMM_KR(   8,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 264,   96, _, _)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, _, _)
    TBLIS_CONFIG_GEMM_KC( 128,  192, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_4x24,
                          bli_dgemm_asm_4x12,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, _, _)

    TBLIS_CONFIG_CHECK(haswell_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(haswell_d8x6)

    TBLIS_CONFIG_GEMM_MR(  16,    8, _, _)
    TBLIS_CONFIG_GEMM_NR(   6,    6, _, _)
    TBLIS_CONFIG_GEMM_KR(   8,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 144,   72, _, _)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, _, _)
    TBLIS_CONFIG_GEMM_KC( 256,  256, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_16x6,
                          bli_dgemm_asm_8x6,
                          _,
                          _)

    TBLIS_CONFIG_CHECK(haswell_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(haswell_d6x8)

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

    TBLIS_CONFIG_CHECK(haswell_check)

TBLIS_END_CONFIG

typedef haswell_d6x8_config haswell_config;

}

#endif
