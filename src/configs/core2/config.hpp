#ifndef _TBLIS_CONFIGS_CORE2_CONFIG_HPP_
#define _TBLIS_CONFIGS_CORE2_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C" tblis::gemm_ukr_func<          float> bli_sgemm_asm_8x4;
extern "C" tblis::gemm_ukr_func<         double> bli_dgemm_asm_4x4;
extern "C" tblis::gemm_ukr_func<tblis::scomplex> bli_cgemm_asm_4x2;
extern "C" tblis::gemm_ukr_func<tblis::dcomplex> bli_zgemm_asm_2x2;

namespace tblis
{

TBLIS_CONFIG(core2);

TBLIS_CONFIG_UKR   (core2, float, gemm, bli_sgemm_asm_8x4);
TBLIS_CONFIG_BS_DEF(core2, float, gemm, mc, 768);
TBLIS_CONFIG_BS_DEF(core2, float, gemm, kc, 384);
TBLIS_CONFIG_BS_DEF(core2, float, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(core2, float, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(core2, float, gemm, nr, 4);

TBLIS_CONFIG_UKR   (core2, double, gemm, bli_dgemm_asm_4x4);
TBLIS_CONFIG_BS_DEF(core2, double, gemm, mc, 384);
TBLIS_CONFIG_BS_DEF(core2, double, gemm, kc, 384);
TBLIS_CONFIG_BS_DEF(core2, double, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(core2, double, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(core2, double, gemm, nr, 4);

TBLIS_CONFIG_UKR   (core2, scomplex, gemm, bli_cgemm_asm_4x2);
TBLIS_CONFIG_BS_DEF(core2, scomplex, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(core2, scomplex, gemm, nr, 2);

TBLIS_CONFIG_UKR   (core2, dcomplex, gemm, bli_zgemm_asm_2x2);
TBLIS_CONFIG_BS_DEF(core2, dcomplex, gemm, mr, 2);
TBLIS_CONFIG_BS_DEF(core2, dcomplex, gemm, nr, 2);

}

#endif
