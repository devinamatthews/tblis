#ifndef _TBLIS_CONFIGS_BULLDOZER_CONFIG_HPP_
#define _TBLIS_CONFIGS_BULLDOZER_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C" tblis::gemm_ukr_func<          float> bli_sgemm_asm_8x8_fma4;
extern "C" tblis::gemm_ukr_func<         double> bli_dgemm_asm_4x6_fma4;
extern "C" tblis::gemm_ukr_func<tblis::scomplex> bli_cgemm_asm_8x4_fma4;
extern "C" tblis::gemm_ukr_func<tblis::dcomplex> bli_zgemm_asm_4x4_fma4;

namespace tblis
{

TBLIS_CONFIG(bulldozer);

TBLIS_CONFIG_UKR   (bulldozer, float, gemm, bli_sgemm_asm_8x8_fma4);
TBLIS_CONFIG_BS_DEF(bulldozer, float, gemm, mc, 128);
TBLIS_CONFIG_BS_DEF(bulldozer, float, gemm, kc, 384);
TBLIS_CONFIG_BS_DEF(bulldozer, float, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(bulldozer, float, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(bulldozer, float, gemm, nr, 8);

TBLIS_CONFIG_UKR   (bulldozer, double, gemm, bli_dgemm_asm_4x6_fma4);
TBLIS_CONFIG_BS_DEF(bulldozer, double, gemm, mc, 1080);
TBLIS_CONFIG_BS_DEF(bulldozer, double, gemm, kc, 120);
TBLIS_CONFIG_BS_DEF(bulldozer, double, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(bulldozer, double, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(bulldozer, double, gemm, nr, 6);

TBLIS_CONFIG_UKR   (bulldozer, scomplex, gemm, bli_cgemm_asm_8x4_fma4);
TBLIS_CONFIG_BS_DEF(bulldozer, scomplex, gemm, mc, 96);
TBLIS_CONFIG_BS_DEF(bulldozer, scomplex, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(bulldozer, scomplex, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(bulldozer, scomplex, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(bulldozer, scomplex, gemm, nr, 4);

TBLIS_CONFIG_UKR   (bulldozer, dcomplex, gemm, bli_zgemm_asm_4x4_fma4);
TBLIS_CONFIG_BS_DEF(bulldozer, dcomplex, gemm, mc, 64);
TBLIS_CONFIG_BS_DEF(bulldozer, dcomplex, gemm, kc, 192);
TBLIS_CONFIG_BS_DEF(bulldozer, dcomplex, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(bulldozer, dcomplex, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(bulldozer, dcomplex, gemm, nr, 4);

}

#endif
