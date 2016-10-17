#ifndef _TBLIS_CONFIGS_SANDYBRIDGE_CONFIG_HPP_
#define _TBLIS_CONFIGS_SANDYBRIDGE_CONFIG_HPP_

#include "configs/configs.hpp"

extern tblis::gemm_ukr_t<          float> bli_sgemm_asm_8x8;
extern tblis::gemm_ukr_t<         double> bli_dgemm_asm_8x4;
extern tblis::gemm_ukr_t<tblis::scomplex> bli_cgemm_asm_8x4;
extern tblis::gemm_ukr_t<tblis::dcomplex> bli_zgemm_asm_4x4;

namespace tblis
{

TBLIS_CONFIG(sandybridge_config);

TBLIS_CONFIG_GEMM_UKR(sandybridge_config, float, bli_sgemm_asm_8x8);
TBLIS_CONFIG_MC(sandybridge_config, float, 128);
TBLIS_CONFIG_KC(sandybridge_config, float, 384);
TBLIS_CONFIG_NC(sandybridge_config, float, 4096);
TBLIS_CONFIG_MR(sandybridge_config, float, 8);
TBLIS_CONFIG_NR(sandybridge_config, float, 8);

TBLIS_CONFIG_GEMM_UKR(sandybridge_config, double, bli_dgemm_asm_8x4);
TBLIS_CONFIG_MC(sandybridge_config, double, 96);
TBLIS_CONFIG_KC(sandybridge_config, double, 256);
TBLIS_CONFIG_NC(sandybridge_config, double, 4096);
TBLIS_CONFIG_MR(sandybridge_config, double, 8);
TBLIS_CONFIG_NR(sandybridge_config, double, 4);

TBLIS_CONFIG_GEMM_UKR(sandybridge_config, scomplex, bli_cgemm_asm_8x4);
TBLIS_CONFIG_MC(sandybridge_config, scomplex, 96);
TBLIS_CONFIG_KC(sandybridge_config, scomplex, 256);
TBLIS_CONFIG_NC(sandybridge_config, scomplex, 4096);
TBLIS_CONFIG_MR(sandybridge_config, scomplex, 8);
TBLIS_CONFIG_NR(sandybridge_config, scomplex, 4);

TBLIS_CONFIG_GEMM_UKR(sandybridge_config, dcomplex, bli_zgemm_asm_4x4);
TBLIS_CONFIG_MC(sandybridge_config, dcomplex, 64);
TBLIS_CONFIG_KC(sandybridge_config, dcomplex, 192);
TBLIS_CONFIG_NC(sandybridge_config, dcomplex, 4096);
TBLIS_CONFIG_MR(sandybridge_config, dcomplex, 4);
TBLIS_CONFIG_NR(sandybridge_config, dcomplex, 4);

}

#endif
