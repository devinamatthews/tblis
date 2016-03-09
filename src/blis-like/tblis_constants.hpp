#ifndef _TBLIS_CONSTANTS_HPP_
#define _TBLIS_CONSTANTS_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T> struct MC {};
template <> struct MC<   float> { static constexpr dim_t def = BLIS_DEFAULT_MC_S, max = BLIS_MAXIMUM_MC_S; };
template <> struct MC<  double> { static constexpr dim_t def = BLIS_DEFAULT_MC_D, max = BLIS_MAXIMUM_MC_D; };
template <> struct MC<sComplex> { static constexpr dim_t def = BLIS_DEFAULT_MC_C, max = BLIS_MAXIMUM_MC_C; };
template <> struct MC<dComplex> { static constexpr dim_t def = BLIS_DEFAULT_MC_Z, max = BLIS_MAXIMUM_MC_Z; };

template <typename T> struct NC {};
template <> struct NC<   float> { static constexpr dim_t def = BLIS_DEFAULT_NC_S, max = BLIS_MAXIMUM_NC_S; };
template <> struct NC<  double> { static constexpr dim_t def = BLIS_DEFAULT_NC_D, max = BLIS_MAXIMUM_NC_D; };
template <> struct NC<sComplex> { static constexpr dim_t def = BLIS_DEFAULT_NC_C, max = BLIS_MAXIMUM_NC_C; };
template <> struct NC<dComplex> { static constexpr dim_t def = BLIS_DEFAULT_NC_Z, max = BLIS_MAXIMUM_NC_Z; };

template <typename T> struct KC {};
template <> struct KC<   float> { static constexpr dim_t def = BLIS_DEFAULT_KC_S, max = BLIS_MAXIMUM_KC_S; };
template <> struct KC<  double> { static constexpr dim_t def = BLIS_DEFAULT_KC_D, max = BLIS_MAXIMUM_KC_D; };
template <> struct KC<sComplex> { static constexpr dim_t def = BLIS_DEFAULT_KC_C, max = BLIS_MAXIMUM_KC_C; };
template <> struct KC<dComplex> { static constexpr dim_t def = BLIS_DEFAULT_KC_Z, max = BLIS_MAXIMUM_KC_Z; };

template <typename T> struct MR {};
template <> struct MR<   float> { static constexpr dim_t def = BLIS_DEFAULT_MR_S, max = BLIS_DEFAULT_MR_S; };
template <> struct MR<  double> { static constexpr dim_t def = BLIS_DEFAULT_MR_D, max = BLIS_DEFAULT_MR_D; };
template <> struct MR<sComplex> { static constexpr dim_t def = BLIS_DEFAULT_MR_C, max = BLIS_DEFAULT_MR_C; };
template <> struct MR<dComplex> { static constexpr dim_t def = BLIS_DEFAULT_MR_Z, max = BLIS_DEFAULT_MR_Z; };

template <typename T> struct NR {};
template <> struct NR<   float> { static constexpr dim_t def = BLIS_DEFAULT_NR_S, max = BLIS_DEFAULT_NR_S; };
template <> struct NR<  double> { static constexpr dim_t def = BLIS_DEFAULT_NR_D, max = BLIS_DEFAULT_NR_D; };
template <> struct NR<sComplex> { static constexpr dim_t def = BLIS_DEFAULT_NR_C, max = BLIS_DEFAULT_NR_C; };
template <> struct NR<dComplex> { static constexpr dim_t def = BLIS_DEFAULT_NR_Z, max = BLIS_DEFAULT_NR_Z; };

template <typename T> struct KR {};
template <> struct KR<   float> { static constexpr dim_t def = BLIS_DEFAULT_KR_S, max = BLIS_DEFAULT_KR_S; };
template <> struct KR<  double> { static constexpr dim_t def = BLIS_DEFAULT_KR_D, max = BLIS_DEFAULT_KR_D; };
template <> struct KR<sComplex> { static constexpr dim_t def = BLIS_DEFAULT_KR_C, max = BLIS_DEFAULT_KR_C; };
template <> struct KR<dComplex> { static constexpr dim_t def = BLIS_DEFAULT_KR_Z, max = BLIS_DEFAULT_KR_Z; };

template <typename T> struct gemm_ukr_t {};
template <> struct gemm_ukr_t<   float> { static constexpr sgemm_ukr_t value = BLIS_SGEMM_UKERNEL; };
template <> struct gemm_ukr_t<  double> { static constexpr dgemm_ukr_t value = BLIS_DGEMM_UKERNEL; };
template <> struct gemm_ukr_t<sComplex> { static constexpr cgemm_ukr_t value = BLIS_CGEMM_UKERNEL; };
template <> struct gemm_ukr_t<dComplex> { static constexpr zgemm_ukr_t value = BLIS_ZGEMM_UKERNEL; };

template <typename T> struct basic_type { typedef T type; };
template <> struct basic_type<sComplex> { typedef scomplex type; };
template <> struct basic_type<dComplex> { typedef dcomplex type; };
template <typename T> using basic_type_t = typename basic_type<T>::type;

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
}

}
}

#endif
