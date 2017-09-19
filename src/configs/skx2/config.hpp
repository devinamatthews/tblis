#ifndef _TBLIS_CONFIGS_SKX2_CONFIG_HPP_
#define _TBLIS_CONFIGS_SKX2_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR(double, bli_dgemm_opt_12x16_l1);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_12x16_l2);
EXTERN_GEMM_UKR(double, bli_sgemm_opt_12x32_l2);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_8x24_l1);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_8x24_l2);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_6x32_l1);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_6x32_l2);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_24x8_knl);

}

namespace tblis
{

extern int skx2_check();

#define L2_BLOCK_SIZES \
    TBLIS_CONFIG_GEMM_MC( 480,   240, _, _) \
    TBLIS_CONFIG_GEMM_NC(5760,  5760, _, _) \
    TBLIS_CONFIG_GEMM_KC_MAX(384, 384, _, _, \
                             480, 480, _, _) \
    TBLIS_CONFIG_M_THREAD_RATIO(_,3,_,_) \
    TBLIS_CONFIG_N_THREAD_RATIO(_,2,_,_) \
    TBLIS_CONFIG_MR_MAX_THREAD(_,1,_,_) \
    TBLIS_CONFIG_NR_MAX_THREAD(_,4,_,_)


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

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_32x6_l2)

    TBLIS_CONFIG_GEMM_MR(_,    32, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     6, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    L2_BLOCK_SIZES

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_6x32_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_24x8_l1)

    TBLIS_CONFIG_GEMM_MR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   768, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   128, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_24x8_l2)

    TBLIS_CONFIG_GEMM_MR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    L2_BLOCK_SIZES

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_16x12_l1)

    TBLIS_CONFIG_GEMM_MR(_,    16, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    12, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   384, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   128, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_12x16_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_16x12_l2)

    TBLIS_CONFIG_GEMM_MR(32, 16, _, _)
    TBLIS_CONFIG_GEMM_NR(12, 12, _, _)
    TBLIS_CONFIG_GEMM_KR(16,  8, _, _)
    L2_BLOCK_SIZES

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_12x32_l2,
                          bli_dgemm_opt_12x16_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(false, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(true, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_12x16_l1)

    TBLIS_CONFIG_GEMM_MR(_,    12, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    16, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   384, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   128, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_12x16_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

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

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x24_l1)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,    24, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   336, _, _)
    TBLIS_CONFIG_GEMM_NC(_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   128, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x24_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

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

    TBLIS_CONFIG_CHECK(skx2_check)

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

    TBLIS_CONFIG_CHECK(skx2_check)

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

    TBLIS_CONFIG_CHECK(skx2_check)

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

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x8_l1)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   256, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   256, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x8_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x8_l2)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   256, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   256, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x8_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, false, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x8_l1_flip)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   256, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   256, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x8_l1,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(skx_8x8_l2_flip)

    TBLIS_CONFIG_GEMM_MR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR(_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC(_,   256, _, _)
    TBLIS_CONFIG_GEMM_NC(_,  5760, _, _)
    TBLIS_CONFIG_GEMM_KC(_,   256, _, _)

    TBLIS_CONFIG_GEMM_UKR(_,
                          bli_dgemm_opt_8x8_l2,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(_, true, _, _)

    TBLIS_CONFIG_CHECK(skx2_check)

TBLIS_END_CONFIG

typedef skx_16x12_l2_config skx2_config;

}

#endif
