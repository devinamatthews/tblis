#ifndef _TBLIS_CONFIGS_ARMV8A_CONFIG_HPP_
#define _TBLIS_CONFIGS_ARMV8A_CONFIG_HPP_

#include "configs/config_builder.hpp"

EXTERN_BLIS_GEMM_UKR(bli_sgemm_armv8a_asm_8x12);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_armv8a_asm_6x8);

namespace tblis
{

extern int armv8a_check();

TBLIS_BEGIN_CONFIG(armv8a)

    TBLIS_CONFIG_GEMM_MR(   8,    6, _, _)
    TBLIS_CONFIG_GEMM_NR(  12,    8, _, _)
    TBLIS_CONFIG_GEMM_KR(   4,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 120,  120, _, _)
    TBLIS_CONFIG_GEMM_NC(3072, 3072, _, _)
    TBLIS_CONFIG_GEMM_KC( 640,  240, _, _)
                        
    // TBLIS_CONFIG_M_THREAD_RATIO(_,3,_,_)
    // TBLIS_CONFIG_N_THREAD_RATIO(_,2,_,_)
    // TBLIS_CONFIG_MR_MAX_THREAD(_,1,_,_)
    // TBLIS_CONFIG_NR_MAX_THREAD(_,4,_,_)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_armv8a_asm_8x12,
                               bli_dgemm_armv8a_asm_6x8,
                               _,
                               _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(false, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(true, true, _, _)

    TBLIS_CONFIG_CHECK(armv8a_check)

TBLIS_END_CONFIG

}

#endif
