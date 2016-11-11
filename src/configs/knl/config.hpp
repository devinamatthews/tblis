#ifndef _TBLIS_CONFIGS_KNL_CONFIG_HPP_
#define _TBLIS_CONFIGS_KNL_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C" tblis::gemm_ukr_func< float> bli_sgemm_opt_30x16_knc;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_opt_30x8_knc;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_opt_30x8;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_opt_24x8;

namespace tblis
{

extern pack_nn_ukr_func<double> knl_packm_30xk;
extern pack_nn_ukr_func<double> knl_packm_24xk;
extern pack_nn_ukr_func<double> knl_packm_8xk;

TBLIS_CONFIG(knl_d30x8_knc);

TBLIS_CONFIG_UKR          (knl_d30x8_knc, float, gemm, bli_sgemm_opt_30x16_knc);
TBLIS_CONFIG_BS_DEF_MAX   (knl_d30x8_knc, float, gemm, mc, 240, 300);
TBLIS_CONFIG_BS_DEF_MAX   (knl_d30x8_knc, float, gemm, kc, 240, 300);
TBLIS_CONFIG_BS_DEF       (knl_d30x8_knc, float, gemm, nc, 14400);
TBLIS_CONFIG_BS_DEF_EXTENT(knl_d30x8_knc, float, gemm, mr, 30, 32);
TBLIS_CONFIG_BS_DEF       (knl_d30x8_knc, float, gemm, nr, 16);
TBLIS_CONFIG_ROW_MAJOR    (knl_d30x8_knc, float, gemm);

TBLIS_CONFIG_UKR          (knl_d30x8_knc, double, gemm, bli_dgemm_opt_30x8_knc);
TBLIS_CONFIG_BS_DEF_MAX   (knl_d30x8_knc, double, gemm, mc, 120, 150);
TBLIS_CONFIG_BS_DEF_MAX   (knl_d30x8_knc, double, gemm, kc, 240, 300);
TBLIS_CONFIG_BS_DEF       (knl_d30x8_knc, double, gemm, nc, 14400);
TBLIS_CONFIG_BS_DEF_EXTENT(knl_d30x8_knc, double, gemm, mr, 30, 32);
TBLIS_CONFIG_BS_DEF       (knl_d30x8_knc, double, gemm, nr, 8);
TBLIS_CONFIG_ROW_MAJOR    (knl_d30x8_knc, double, gemm);

TBLIS_CONFIG(knl_d30x8);

TBLIS_CONFIG_UKR          (knl_d30x8, double, gemm, bli_dgemm_opt_30x8);
TBLIS_CONFIG_BS_DEF_MAX   (knl_d30x8, double, gemm, mc, 120, 150);
TBLIS_CONFIG_BS_DEF_MAX   (knl_d30x8, double, gemm, kc, 240, 300);
TBLIS_CONFIG_BS_DEF       (knl_d30x8, double, gemm, nc, 14400);
TBLIS_CONFIG_BS_DEF_EXTENT(knl_d30x8, double, gemm, mr, 30, 32);
TBLIS_CONFIG_BS_DEF       (knl_d30x8, double, gemm, nr, 8);
TBLIS_CONFIG_ROW_MAJOR    (knl_d30x8, double, gemm);
TBLIS_CONFIG_UKR          (knl_d30x8, double, pack_nn_mr, knl_packm_30xk);
TBLIS_CONFIG_UKR          (knl_d30x8, double, pack_nn_nr, knl_packm_8xk);
TBLIS_CONFIG_BS_DEF       (knl_d30x8, double, gemm, kr, 8);

TBLIS_CONFIG(knl_d24x8);

TBLIS_CONFIG_UKR       (knl_d24x8, double, gemm, bli_dgemm_opt_24x8);
TBLIS_CONFIG_BS_DEF_MAX(knl_d24x8, double, gemm, mc, 120, 150);
TBLIS_CONFIG_BS_DEF_MAX(knl_d24x8, double, gemm, kc, 336, 420);
TBLIS_CONFIG_BS_DEF    (knl_d24x8, double, gemm, nc, 14400);
TBLIS_CONFIG_BS_DEF    (knl_d24x8, double, gemm, mr, 24);
TBLIS_CONFIG_BS_DEF    (knl_d24x8, double, gemm, nr, 8);
TBLIS_CONFIG_ROW_MAJOR (knl_d24x8, double, gemm);
TBLIS_CONFIG_UKR       (knl_d24x8, double, pack_nn_mr, knl_packm_24xk);
TBLIS_CONFIG_UKR       (knl_d24x8, double, pack_nn_nr, knl_packm_8xk);
TBLIS_CONFIG_BS_DEF    (knl_d24x8, double, gemm, kr, 8);

typedef knl_d24x8_config knl_config;

}

#endif
