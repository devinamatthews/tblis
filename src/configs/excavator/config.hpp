#ifndef _TBLIS_CONFIGS_EXCAVATOR_CONFIG_HPP_
#define _TBLIS_CONFIGS_EXCAVATOR_CONFIG_HPP_

#include "configs/config_builder.hpp"

/*
 * These are actually the same kernels as Piledriver.
 */
extern "C" tblis::gemm_ukr_func<          float> bli_sgemm_asm_16x3;
extern "C" tblis::gemm_ukr_func<         double> bli_dgemm_asm_8x3;
extern "C" tblis::gemm_ukr_func<tblis::scomplex> bli_cgemm_asm_4x2;
extern "C" tblis::gemm_ukr_func<tblis::dcomplex> bli_zgemm_asm_2x2;

namespace tblis
{

TBLIS_CONFIG(excavator);

TBLIS_CONFIG_UKR   (excavator, float, gemm, bli_sgemm_asm_16x3);
TBLIS_CONFIG_BS_DEF(excavator, float, gemm, mc, 528);
TBLIS_CONFIG_BS_DEF(excavator, float, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(excavator, float, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(excavator, float, gemm, mr, 16);
TBLIS_CONFIG_BS_DEF(excavator, float, gemm, nr, 3);

TBLIS_CONFIG_UKR   (excavator, double, gemm, bli_dgemm_asm_8x3);
TBLIS_CONFIG_BS_DEF(excavator, double, gemm, mc, 264);
TBLIS_CONFIG_BS_DEF(excavator, double, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(excavator, double, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(excavator, double, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(excavator, double, gemm, nr, 3);

TBLIS_CONFIG_UKR   (excavator, scomplex, gemm, bli_cgemm_asm_4x2);
TBLIS_CONFIG_BS_DEF(excavator, scomplex, gemm, mc, 264);
TBLIS_CONFIG_BS_DEF(excavator, scomplex, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(excavator, scomplex, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(excavator, scomplex, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(excavator, scomplex, gemm, nr, 2);

TBLIS_CONFIG_UKR   (excavator, dcomplex, gemm, bli_zgemm_asm_2x2);
TBLIS_CONFIG_BS_DEF(excavator, dcomplex, gemm, mc, 100);
TBLIS_CONFIG_BS_DEF(excavator, dcomplex, gemm, kc, 320);
TBLIS_CONFIG_BS_DEF(excavator, dcomplex, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(excavator, dcomplex, gemm, mr, 2);
TBLIS_CONFIG_BS_DEF(excavator, dcomplex, gemm, nr, 2);

}

#endif
