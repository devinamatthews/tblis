#ifndef _TBLIS_CONFIG_HPP_
#define _TBLIS_CONFIG_HPP_

#include "tblis.hpp"

extern "C"
{
#include "bli_kernel_pre_macro_defs.h"
}

namespace tblis
{
namespace blis_like
{

template <typename T>
using gemm_ukr_t =
void (*)(dim_t k,
         const T* restrict alpha,
         const T* restrict a, const T* restrict b,
         const T* restrict beta,
         T* restrict c, inc_t rs_c, inc_t cs_c,
         const void* restrict data, const void* restrict cntx);

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
    enum {NT_NONE, NT_IR, NT_JR, NT_IC, NT_JC, NT_KC};
}

}
}

#define TBLIS_CONFIG dunnington
#define TBLIS_CONFIG_NAME Dunnington
#include "tblis_config_import.hpp"

#define TBLIS_CONFIG sandybridge
#define TBLIS_CONFIG_NAME SandyBridge
#include "tblis_config_import.hpp"

#define TBLIS_CONFIG haswell
#define TBLIS_CONFIG_NAME Haswell
#include "tblis_config_import.hpp"

#endif
