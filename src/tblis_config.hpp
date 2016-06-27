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
}

}
}

#define TBLIS_CONFIG dunnington
#include "tblis_config_import.hpp"

#define TBLIS_CONFIG sandybridge
#include "tblis_config_import.hpp"

#define TBLIS_CONFIG haswell
#include "tblis_config_import.hpp"

#endif
