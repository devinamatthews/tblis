#ifndef _TBLIS_CONFIGS_HASWELL_CONFIG_HPP_
#define _TBLIS_CONFIGS_HASWELL_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C" tblis::gemm_ukr_func< float> bli_sgemm_asm_24x4;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_asm_12x4;
extern "C" tblis::gemm_ukr_func< float> bli_sgemm_asm_4x24;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_asm_4x12;
extern "C" tblis::gemm_ukr_func< float> bli_sgemm_asm_16x6;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_asm_8x6;
extern "C" tblis::gemm_ukr_func< float> bli_sgemm_asm_6x16;
extern "C" tblis::gemm_ukr_func<double> bli_dgemm_asm_6x8;

namespace tblis
{

TBLIS_CONFIG(haswell_d12x4);

TBLIS_CONFIG_UKR   (haswell_d12x4, float, gemm, bli_sgemm_asm_24x4);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, float, gemm, mc, 264);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, float, gemm, kc, 128);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, float, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, float, gemm, mr, 24);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, float, gemm, nr, 4);

TBLIS_CONFIG_UKR   (haswell_d12x4, double, gemm, bli_dgemm_asm_12x4);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, double, gemm, mc, 96);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, double, gemm, kc, 192);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, double, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, double, gemm, mr, 12);
TBLIS_CONFIG_BS_DEF(haswell_d12x4, double, gemm, nr, 4);

TBLIS_CONFIG(haswell_d4x12);

TBLIS_CONFIG_UKR   (haswell_d4x12, float, gemm, bli_sgemm_asm_4x24);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, float, gemm, mc, 264);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, float, gemm, kc, 128);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, float, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, float, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, float, gemm, nr, 24);
TBLIS_CONFIG_ROW_MAJOR(haswell_d4x12, float, gemm);

TBLIS_CONFIG_UKR   (haswell_d4x12, double, gemm, bli_dgemm_asm_4x12);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, double, gemm, mc, 96);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, double, gemm, kc, 192);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, double, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, double, gemm, mr, 4);
TBLIS_CONFIG_BS_DEF(haswell_d4x12, double, gemm, nr, 12);
TBLIS_CONFIG_ROW_MAJOR(haswell_d4x12, double, gemm);

TBLIS_CONFIG(haswell_d8x6);

TBLIS_CONFIG_UKR   (haswell_d8x6, float, gemm, bli_sgemm_asm_16x6);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, float, gemm, mc, 144);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, float, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, float, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, float, gemm, mr, 16);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, float, gemm, nr, 6);

TBLIS_CONFIG_UKR   (haswell_d8x6, double, gemm, bli_dgemm_asm_8x6);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, double, gemm, mc, 72);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, double, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, double, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, double, gemm, mr, 8);
TBLIS_CONFIG_BS_DEF(haswell_d8x6, double, gemm, nr, 6);

TBLIS_CONFIG(haswell_d6x8);

TBLIS_CONFIG_UKR   (haswell_d6x8, float, gemm, bli_sgemm_asm_6x16);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, float, gemm, mc, 144);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, float, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, float, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, float, gemm, mr, 6);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, float, gemm, nr, 16);
TBLIS_CONFIG_ROW_MAJOR(haswell_d6x8, float, gemm);

TBLIS_CONFIG_UKR   (haswell_d6x8, double, gemm, bli_dgemm_asm_6x8);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, double, gemm, mc, 72);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, double, gemm, kc, 256);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, double, gemm, nc, 4080);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, double, gemm, mr, 6);
TBLIS_CONFIG_BS_DEF(haswell_d6x8, double, gemm, nr, 8);
TBLIS_CONFIG_ROW_MAJOR(haswell_d6x8, double, gemm);

typedef haswell_d6x8_config haswell_config;

}

#endif
