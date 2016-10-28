#ifndef _TBLIS_CONFIGS_PILEDRIVER_CONFIG_HPP_
#define _TBLIS_CONFIGS_PILEDRIVER_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C" tblis::gemm_ukr_func<          float> bli_sgemm_asm_16x3;
extern "C" tblis::gemm_ukr_func<         double> bli_dgemm_asm_8x3;
extern "C" tblis::gemm_ukr_func<tblis::scomplex> bli_cgemm_asm_4x2;
extern "C" tblis::gemm_ukr_func<tblis::dcomplex> bli_zgemm_asm_2x2;

namespace tblis
{

TBLIS_CONFIG(piledriver);

TBLIS_CONFIG_UKR   (piledriver, float, gemm, bli_sgemm_asm_16x3);
TBLIS_CONFIG_BS_DEF(piledriver, float, gemm, mc, 2016);
TBLIS_CONFIG_BS_DEF(piledriver, float, gemm, kc, 128);
TBLIS_CONFIG_BS_DEF(piledriver, float, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(piledriver, float, gemm, mr, 16);
TBLIS_CONFIG_BS_DEF(piledriver, float, gemm, nr, 3);

TBLIS_CONFIG_UKR   (piledriver, double, gemm, bli_dgemm_asm_8x3);
TBLIS_CONFIG_BS_DEF(piledriver, double, gemm, mc, 1008);
TBLIS_CONFIG_BS_DEF(piledriver, double, gemm, kc, 128);
TBLIS_CONFIG_BS_DEF(piledriver, double, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(piledriver, double, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(piledriver, double, gemm, nr, 3);

TBLIS_CONFIG_UKR   (piledriver, scomplex, gemm, bli_cgemm_asm_4x2);
TBLIS_CONFIG_BS_DEF(piledriver, scomplex, gemm, mc, 512);
TBLIS_CONFIG_BS_DEF(piledriver, scomplex, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(piledriver, scomplex, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(piledriver, scomplex, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(piledriver, scomplex, gemm, nr, 2);

TBLIS_CONFIG_UKR   (piledriver, dcomplex, gemm, bli_zgemm_asm_2x2);
TBLIS_CONFIG_BS_DEF(piledriver, dcomplex, gemm, mc, 400);
TBLIS_CONFIG_BS_DEF(piledriver, dcomplex, gemm, kc, 160);
TBLIS_CONFIG_BS_DEF(piledriver, dcomplex, gemm, nc, 8400);
TBLIS_CONFIG_BS_DEF(piledriver, dcomplex, gemm, mr, 2);
TBLIS_CONFIG_BS_DEF(piledriver, dcomplex, gemm, nr, 2);

}

#endif
