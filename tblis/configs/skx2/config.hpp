#ifndef _TBLIS_CONFIGS_SKX2_CONFIG_HPP_
#define _TBLIS_CONFIGS_SKX2_CONFIG_HPP_

#include "configs/config_builder.hpp"

EXTERN_BLIS_GEMM_UKR(bli_sgemm_opt_12x32_l2);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_opt_12x16_l2);

namespace tblis
{

extern int skx2_check();

TBLIS_BEGIN_CONFIG(skx2)

    TBLIS_CONFIG_GEMM_MR(32, 16, _, _)
    TBLIS_CONFIG_GEMM_NR(12, 12, _, _)
    TBLIS_CONFIG_GEMM_KR(16,  8, _, _)
    TBLIS_CONFIG_GEMM_MC( 480,   240, _, _)
    TBLIS_CONFIG_GEMM_NC(3072,  3072, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX(384, 384, _, _,
                             480, 480, _, _)
    TBLIS_CONFIG_M_THREAD_RATIO(_,3,_,_)
    TBLIS_CONFIG_N_THREAD_RATIO(_,2,_,_)
    TBLIS_CONFIG_MR_MAX_THREAD(_,1,_,_)
    TBLIS_CONFIG_NR_MAX_THREAD(_,4,_,_)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_opt_12x32_l2,
                               bli_dgemm_opt_12x16_l2,
                               _,
                               _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(false, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(true, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

}

#endif
