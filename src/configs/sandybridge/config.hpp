#ifndef _TBLIS_CONFIGS_SANDYBRIDGE_CONFIG_HPP_
#define _TBLIS_CONFIGS_SANDYBRIDGE_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C" tblis::gemm_ukr_func<          float> bli_sgemm_asm_8x8;
extern "C" tblis::gemm_ukr_func<         double> bli_dgemm_asm_8x4;
extern "C" tblis::gemm_ukr_func<tblis::scomplex> bli_cgemm_asm_8x4;
extern "C" tblis::gemm_ukr_func<tblis::dcomplex> bli_zgemm_asm_4x4;

namespace tblis
{

TBLIS_CONFIG(sandybridge);

TBLIS_CONFIG_UKR   (sandybridge, float, gemm, bli_sgemm_asm_8x8);
TBLIS_CONFIG_BS_DEF(sandybridge, float, gemm, mc, 128);
TBLIS_CONFIG_BS_DEF(sandybridge, float, gemm, kc, 384);
TBLIS_CONFIG_BS_DEF(sandybridge, float, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(sandybridge, float, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(sandybridge, float, gemm, nr, 8);

TBLIS_CONFIG_UKR   (sandybridge, double, gemm, bli_dgemm_asm_8x4);
TBLIS_CONFIG_BS_DEF(sandybridge, double, gemm, mc, 96);
TBLIS_CONFIG_BS_DEF(sandybridge, double, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(sandybridge, double, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(sandybridge, double, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(sandybridge, double, gemm, nr, 4);

TBLIS_CONFIG_UKR   (sandybridge, scomplex, gemm, bli_cgemm_asm_8x4);
TBLIS_CONFIG_BS_DEF(sandybridge, scomplex, gemm, mc, 96);
TBLIS_CONFIG_BS_DEF(sandybridge, scomplex, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(sandybridge, scomplex, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(sandybridge, scomplex, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(sandybridge, scomplex, gemm, nr, 4);

TBLIS_CONFIG_UKR   (sandybridge, dcomplex, gemm, bli_zgemm_asm_4x4);
TBLIS_CONFIG_BS_DEF(sandybridge, dcomplex, gemm, mc, 64);
TBLIS_CONFIG_BS_DEF(sandybridge, dcomplex, gemm, kc, 192);
TBLIS_CONFIG_BS_DEF(sandybridge, dcomplex, gemm, nc, 4096);
TBLIS_CONFIG_BS_DEF(sandybridge, dcomplex, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(sandybridge, dcomplex, gemm, nr, 4);

}

#endif
