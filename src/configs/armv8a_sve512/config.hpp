#ifndef _TBLIS_CONFIGS_ARMV8A_SVE512_CONFIG_HPP_
#define _TBLIS_CONFIGS_ARMV8A_SVE512_CONFIG_HPP_

#include "configs/config_builder.hpp"

EXTERN_BLIS_GEMM_UKR(bli_sgemm_armsve_asm_2vx10_unindexed);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_armsve_asm_2vx10_unindexed);

namespace tblis
{

EXTERN_PACK_NN_UKR(double, sve512_dpackm_asm_16xk);
EXTERN_PACK_NN_UKR(double, sve512_dpackm_asm_10xk);

extern int armv8a_sve512_check();

TBLIS_BEGIN_CONFIG(armv8a_sve512)

    TBLIS_CONFIG_GEMM_MR(   32,   16, _, _)
    TBLIS_CONFIG_GEMM_NR(   10,   10, _, _)
    TBLIS_CONFIG_GEMM_KR(    4,    4, _, _)
    TBLIS_CONFIG_GEMM_MC(  256,  128, _, _)
    TBLIS_CONFIG_GEMM_NC(23040,23040, _, _)
    TBLIS_CONFIG_GEMM_KC( 2048, 2048, _, _)
                        
    // TBLIS_CONFIG_M_THREAD_RATIO(_,3,_,_)
    // TBLIS_CONFIG_N_THREAD_RATIO(_,2,_,_)
    // TBLIS_CONFIG_MR_MAX_THREAD(_,1,_,_)
    // TBLIS_CONFIG_NR_MAX_THREAD(_,4,_,_)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_armsve_asm_2vx10_unindexed,
                               bli_dgemm_armsve_asm_2vx10_unindexed,
                               _,
                               _)
    // TBLIS_CONFIG_PACK_NN_MR_UKR(_, sve512_dpackm_asm_16xk, _, _)
    // TBLIS_CONFIG_PACK_NN_NR_UKR(_, sve512_dpackm_asm_10xk, _, _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(false, false, _, _)
    TBLIS_CONFIG_GEMM_FLIP_UKR(true, true, _, _)

    TBLIS_CONFIG_CHECK(armv8a_sve512_check)

TBLIS_END_CONFIG

}

#endif
