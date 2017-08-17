#ifndef _TBLIS_CONFIGS_SKX_CONFIG_HPP_
#define _TBLIS_CONFIGS_SKX_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR(double, bli_dgemm_opt_12x16_l1);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_12x16_l2);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_8x24_l1);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_8x24_l2);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_6x32_l1);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_6x32_l2);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_24x8_knl);

}

namespace tblis
{

extern int skx_check();

TBLIS_BEGIN_CONFIG(skx_32x6_l1)

    TBLIS_CONFIG_GEMM_MR(_,    32, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     6, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   320, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_6x32_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_32x6_l2)

    TBLIS_CONFIG_GEMM_MR(_,    32, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     6, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   128, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   336, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_6x32_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_24x8_l1)

    TBLIS_CONFIG_GEMM_MR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   336, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_24x8_l2)

    TBLIS_CONFIG_GEMM_MR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   144, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   336, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_16x12_l1)

    TBLIS_CONFIG_GEMM_MR(_,    16, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    12, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   336, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_12x16_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_16x12_l2)

    TBLIS_CONFIG_GEMM_MR(_,    16, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    12, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   144, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   336, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_12x16_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_12x16_l1)

    TBLIS_CONFIG_GEMM_MR(_,    12, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    16, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   336, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_12x16_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_12x16_l2)

    TBLIS_CONFIG_GEMM_MR(_,    12, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    16, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   144, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   336, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_12x16_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x24_l1)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   336, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x24_l2)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   144, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_6x32_l1)

    TBLIS_CONFIG_GEMM_MR(_,     6, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    32, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   336, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,    96, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_6x32_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_6x32_l2)

    TBLIS_CONFIG_GEMM_MR(_,     6, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    32, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   144, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   336, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_6x32_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_knl)

    TBLIS_CONFIG_GEMM_MR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   120, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   336, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_24x8_knl,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx_check)

TBLIS_END_CONFIG

typedef skx_knl_config skx_config;

}

#endif
