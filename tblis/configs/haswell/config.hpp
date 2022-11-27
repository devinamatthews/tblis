#ifndef TBLIS_CONFIGS_HASWELL_CONFIG_HPP
#define TBLIS_CONFIGS_HASWELL_CONFIG_HPP

#include <tblis/internal/configs.hpp>

namespace tblis
{

EXTERN_BLIS_GEMM_UKR(bli_sgemm_haswell_asm_6x16);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_haswell_asm_6x8);
EXTERN_BLIS_GEMM_UKR(bli_cgemm_haswell_asm_3x8);
EXTERN_BLIS_GEMM_UKR(bli_zgemm_haswell_asm_3x4);

extern int haswell_check();

TBLIS_BEGIN_CONFIG(haswell)

    TBLIS_CONFIG_GEMM_MR(   6,    6,    3,    3)
    TBLIS_CONFIG_GEMM_NR(  16,    8,    8,    4)
    TBLIS_CONFIG_GEMM_KR(   8,    4,    4,    4)
    TBLIS_CONFIG_GEMM_MC( 168,   72,   75,  192)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, 4080, 4080)
    TBLIS_CONFIG_GEMM_KC( 256,  256,  256,  256)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_haswell_asm_6x16,
                               bli_dgemm_haswell_asm_6x8,
                               bli_cgemm_haswell_asm_3x8,
                               bli_zgemm_haswell_asm_3x4)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, true, true)

    TBLIS_CONFIG_CHECK(haswell_check)

TBLIS_END_CONFIG

}

#endif
