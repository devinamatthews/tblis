#ifndef TBLIS_CONFIGS_KNL_CONFIG_HPP
#define TBLIS_CONFIGS_KNL_CONFIG_HPP

#include <tblis/internal/configs.hpp>

namespace tblis
{

EXTERN_BLIS_GEMM_UKR(bli_sgemm_opt_24x16);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_opt_24x8);

EXTERN_PACK_NN_UKR( float, knl_spackm_24xk);
EXTERN_PACK_NN_UKR(double, knl_dpackm_24xk);
EXTERN_PACK_NN_UKR( float, knl_spackm_16xk);
EXTERN_PACK_NN_UKR(double, knl_dpackm_8xk);

extern int knl_check();

TBLIS_BEGIN_CONFIG(knl)

    TBLIS_CONFIG_GEMM_MR    (   24,    24, _, _)
    TBLIS_CONFIG_GEMM_NR    (   16,     8, _, _)
    TBLIS_CONFIG_GEMM_KR    (   16,     8, _, _)
    TBLIS_CONFIG_GEMM_MC    (  240,   120, _, _)
    TBLIS_CONFIG_GEMM_NC    (14400, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX(  336,   336, _, _,
                               408,   408, _, _)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_opt_24x16, bli_dgemm_opt_24x8 , _, _)

    TBLIS_CONFIG_PACK_NN_MR_UKR(knl_spackm_24xk, knl_dpackm_24xk, _, _)
    TBLIS_CONFIG_PACK_NN_NR_UKR(knl_spackm_16xk, knl_dpackm_8xk , _, _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, _, _)

    TBLIS_CONFIG_M_THREAD_RATIO(4, 4, _, _)
    TBLIS_CONFIG_NR_MAX_THREAD(1, 1, _, _)

    TBLIS_CONFIG_CHECK(knl_check)

TBLIS_END_CONFIG

}

#endif
