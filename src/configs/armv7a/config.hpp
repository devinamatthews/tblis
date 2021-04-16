#ifndef _TBLIS_CONFIGS_ARMV7A_CONFIG_HPP_
#define _TBLIS_CONFIGS_ARMV7A_CONFIG_HPP_

#include "configs/config_builder.hpp"

EXTERN_BLIS_GEMM_UKR(bli_sgemm_armv7a_int_4x4);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_armv7a_int_4x4);

namespace tblis
{

extern int armv7a_check();

TBLIS_BEGIN_CONFIG(armv7a)

    TBLIS_CONFIG_GEMM_MR(   4,    4, _, _)
    TBLIS_CONFIG_GEMM_NR(   4,    4, _, _)
    TBLIS_CONFIG_GEMM_KR(   4,    4, _, _)
    TBLIS_CONFIG_GEMM_MC( 336,  176, _, _)
    TBLIS_CONFIG_GEMM_NC(4096, 4096, _, _)
    TBLIS_CONFIG_GEMM_KC( 528,  368, _, _)

    // TBLIS_CONFIG_M_THREAD_RATIO(_,3,_,_)
    // TBLIS_CONFIG_N_THREAD_RATIO(_,2,_,_)
    // TBLIS_CONFIG_MR_MAX_THREAD(_,1,_,_)
    // TBLIS_CONFIG_NR_MAX_THREAD(_,4,_,_)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_armv7a_int_4x4,
                               bli_dgemm_armv7a_int_4x4,
                               _,
                               _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(false, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(true, true, _, _)

    TBLIS_CONFIG_CHECK(armv7a_check)

TBLIS_END_CONFIG

}

#endif
