#ifndef _TBLIS_CONFIGS_KNM_CONFIG_HPP_
#define _TBLIS_CONFIGS_KNM_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR( float, bli_sgemm_opt_24x16);
EXTERN_GEMM_UKR( float, bli_sgemm_opt_16x24);

}

namespace tblis
{

EXTERN_PACK_NN_UKR( float, knm_spackm_24xk);

extern int knm_check();

TBLIS_BEGIN_CONFIG(knm_s24x16)

    TBLIS_CONFIG_GEMM_MR    (   24, _, _, _)
    TBLIS_CONFIG_GEMM_NR    (   16, _, _, _)
    TBLIS_CONFIG_GEMM_KR    (    4, _, _, _)
    TBLIS_CONFIG_GEMM_MC    (  240, _, _, _)
    TBLIS_CONFIG_GEMM_NC    (14400, _, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX(  336, _, _, _,
                               408, _, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_24x16, _ , _, _)

    TBLIS_CONFIG_PACK_NN_MR_UKR(knm_spackm_24xk, _, _, _)
    TBLIS_CONFIG_PACK_NN_NR_UKR(_, _ , _, _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, _, _, _)

    TBLIS_CONFIG_M_THREAD_RATIO(4, _, _, _)
    TBLIS_CONFIG_NR_MAX_THREAD(4, _, _, _)

    TBLIS_CONFIG_CHECK(knm_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(knm_s16x24)

    TBLIS_CONFIG_GEMM_MR    (   16, _, _, _)
    TBLIS_CONFIG_GEMM_NR    (   24, _, _, _)
    TBLIS_CONFIG_GEMM_KR    (    4, _, _, _)
    TBLIS_CONFIG_GEMM_MC    (  240, _, _, _)
    TBLIS_CONFIG_GEMM_NC    (14400, _, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX(  336, _, _, _,
                               408, _, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_16x24, _ , _, _)

    TBLIS_CONFIG_PACK_NN_MR_UKR(_, _, _, _)
    TBLIS_CONFIG_PACK_NN_NR_UKR(knm_spackm_24xk, _ , _, _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(false, _, _, _)

    TBLIS_CONFIG_M_THREAD_RATIO(4, _, _, _)
    TBLIS_CONFIG_NR_MAX_THREAD(4, _, _, _)

    TBLIS_CONFIG_CHECK(knm_check)

TBLIS_END_CONFIG

typedef knm_s24x16_config knm_config;

}

#endif
