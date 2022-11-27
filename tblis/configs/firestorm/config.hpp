#ifndef TBLIS_CONFIGS_FIRESTORM_CONFIG_HPP
#define TBLIS_CONFIGS_FIRESTORM_CONFIG_HPP

#include <tblis/internal/configs.hpp>

namespace tblis
{

EXTERN_BLIS_GEMM_UKR(bli_sgemm_armv8a_asm_12x8r);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_armv8a_asm_8x6r);

extern int firestorm_check();

TBLIS_BEGIN_CONFIG(firestorm)

TBLIS_CONFIG_GEMM_MR(  12,    8, _, _)
TBLIS_CONFIG_GEMM_NR(   8,    6, _, _)
TBLIS_CONFIG_GEMM_KR(   8,    4, _, _)
TBLIS_CONFIG_GEMM_MC( 480,  256, _, _)
TBLIS_CONFIG_GEMM_NC(9600, 8184, _, _)
TBLIS_CONFIG_GEMM_KC(4096, 3072, _, _)

TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_armv8a_asm_12x8r,
                           bli_dgemm_armv8a_asm_8x6r,
                           _,
                           _)

TBLIS_CONFIG_CHECK(firestorm_check)

TBLIS_END_CONFIG

}

#endif
