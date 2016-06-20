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
         T* alpha, T* a, T* b,
         T*  beta, T* c, inc_t rs_c, inc_t cs_c,
         void* data, void* cntx);

template <typename T> struct basic_type { typedef T type; };
template <> struct basic_type<scomplex> { typedef scomplex type; };
template <> struct basic_type<dcomplex> { typedef dcomplex type; };
template <typename T> using basic_type_t = typename basic_type<T>::type;

template <typename T> struct blis_type {};
template <> struct blis_type<   float> { static constexpr num_t value = BLIS_FLOAT; };
template <> struct blis_type<  double> { static constexpr num_t value = BLIS_DOUBLE; };
template <> struct blis_type<scomplex> { static constexpr num_t value = BLIS_SCOMPLEX; };
template <> struct blis_type<dcomplex> { static constexpr num_t value = BLIS_DCOMPLEX; };

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
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
