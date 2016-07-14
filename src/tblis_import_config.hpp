#include "tblis.hpp"

#define TBLIS_CONFIG_HEADER(dir) TBLIS_STRINGIZE(config/dir/bli_kernel.h)
#include TBLIS_CONFIG_HEADER(TBLIS_CONFIG)
#undef TBLIS_CONFIG_HEADER

extern "C"
{

#ifndef BLIS_SGEMM_UKERNEL
#define BLIS_SGEMM_UKERNEL GenericMicroKernel<float,MR<float>::def,NR<float>::def>
#else
void BLIS_SGEMM_UKERNEL(tblis::stride_type k,
                        const float* alpha, const float* p_a, const float* p_b,
                        const float* beta, float* p_c, tblis::stride_type rs_c, tblis::stride_type cs_c,
                        const void* data, const void* ctx);
#endif

#ifndef BLIS_DGEMM_UKERNEL
#define BLIS_DGEMM_UKERNEL GenericMicroKernel<double,MR<double>::def,NR<double>::def>
#else
void BLIS_DGEMM_UKERNEL(tblis::stride_type k,
                        const double* alpha, const double* p_a, const double* p_b,
                        const double* beta, double* p_c, tblis::stride_type rs_c, tblis::stride_type cs_c,
                        const void* data, const void* ctx);
#endif

#ifndef BLIS_CGEMM_UKERNEL
#define BLIS_CGEMM_UKERNEL GenericMicroKernel<tblis::scomplex,MR<tblis::scomplex>::def,NR<tblis::scomplex>::def>
#else
void BLIS_CGEMM_UKERNEL(tblis::stride_type k,
                        const tblis::scomplex* alpha, const tblis::scomplex* p_a, const tblis::scomplex* p_b,
                        const tblis::scomplex* beta, tblis::scomplex* p_c, tblis::stride_type rs_c, tblis::stride_type cs_c,
                        const void* data, const void* ctx);
#endif

#ifndef BLIS_ZGEMM_UKERNEL
#define BLIS_ZGEMM_UKERNEL GenericMicroKernel<tblis::dcomplex,MR<tblis::dcomplex>::def,NR<tblis::dcomplex>::def>
#else
void BLIS_ZGEMM_UKERNEL(tblis::stride_type k,
                        const tblis::dcomplex* alpha, const tblis::dcomplex* p_a, const tblis::dcomplex* p_b,
                        const tblis::dcomplex* beta, tblis::dcomplex* p_c, tblis::stride_type rs_c, tblis::stride_type cs_c,
                        const void* data, const void* ctx);
#endif

}

#include "bli_kernel_macro_defs.h"

namespace tblis
{

#define TBLIS_CONFIG_STRUCT TBLIS_CONCAT(TBLIS_CONFIG, _config)

struct TBLIS_CONFIG_STRUCT
{
    template <typename T> struct MC {};
    template <typename T> struct NC {};
    template <typename T> struct KC {};
    template <typename T> struct MR {};
    template <typename T> struct NR {};
    template <typename T> struct KR {};
    template <typename T> struct gemm_ukr {};
    template <typename T> struct gemm_row_major {};

#ifdef BLIS_TREE_BARRIER
    constexpr static int tree_barrier_arity = BLIS_TREE_BARRIER_ARITY;
#else
    constexpr static int tree_barrier_arity = 0;
#endif
};

template <> struct TBLIS_CONFIG_STRUCT::MC<   float> { static constexpr idx_type def = BLIS_DEFAULT_MC_S, max = BLIS_MAXIMUM_MC_S, iota = BLIS_DEFAULT_MR_S; };
template <> struct TBLIS_CONFIG_STRUCT::MC<  double> { static constexpr idx_type def = BLIS_DEFAULT_MC_D, max = BLIS_MAXIMUM_MC_D, iota = BLIS_DEFAULT_MR_D; };
template <> struct TBLIS_CONFIG_STRUCT::MC<scomplex> { static constexpr idx_type def = BLIS_DEFAULT_MC_C, max = BLIS_MAXIMUM_MC_C, iota = BLIS_DEFAULT_MR_C; };
template <> struct TBLIS_CONFIG_STRUCT::MC<dcomplex> { static constexpr idx_type def = BLIS_DEFAULT_MC_Z, max = BLIS_MAXIMUM_MC_Z, iota = BLIS_DEFAULT_MR_Z; };

template <> struct TBLIS_CONFIG_STRUCT::NC<   float> { static constexpr idx_type def = BLIS_DEFAULT_NC_S, max = BLIS_MAXIMUM_NC_S, iota = BLIS_DEFAULT_NR_S; };
template <> struct TBLIS_CONFIG_STRUCT::NC<  double> { static constexpr idx_type def = BLIS_DEFAULT_NC_D, max = BLIS_MAXIMUM_NC_D, iota = BLIS_DEFAULT_NR_D; };
template <> struct TBLIS_CONFIG_STRUCT::NC<scomplex> { static constexpr idx_type def = BLIS_DEFAULT_NC_C, max = BLIS_MAXIMUM_NC_C, iota = BLIS_DEFAULT_NR_C; };
template <> struct TBLIS_CONFIG_STRUCT::NC<dcomplex> { static constexpr idx_type def = BLIS_DEFAULT_NC_Z, max = BLIS_MAXIMUM_NC_Z, iota = BLIS_DEFAULT_NR_Z; };

template <> struct TBLIS_CONFIG_STRUCT::KC<   float> { static constexpr idx_type def = BLIS_DEFAULT_KC_S, max = BLIS_MAXIMUM_KC_S, iota = BLIS_DEFAULT_KR_S; };
template <> struct TBLIS_CONFIG_STRUCT::KC<  double> { static constexpr idx_type def = BLIS_DEFAULT_KC_D, max = BLIS_MAXIMUM_KC_D, iota = BLIS_DEFAULT_KR_D; };
template <> struct TBLIS_CONFIG_STRUCT::KC<scomplex> { static constexpr idx_type def = BLIS_DEFAULT_KC_C, max = BLIS_MAXIMUM_KC_C, iota = BLIS_DEFAULT_KR_C; };
template <> struct TBLIS_CONFIG_STRUCT::KC<dcomplex> { static constexpr idx_type def = BLIS_DEFAULT_KC_Z, max = BLIS_MAXIMUM_KC_Z, iota = BLIS_DEFAULT_KR_Z; };

template <> struct TBLIS_CONFIG_STRUCT::MR<   float> { static constexpr idx_type def = BLIS_DEFAULT_MR_S, extent = BLIS_PACKDIM_MR_S; };
template <> struct TBLIS_CONFIG_STRUCT::MR<  double> { static constexpr idx_type def = BLIS_DEFAULT_MR_D, extent = BLIS_PACKDIM_MR_D; };
template <> struct TBLIS_CONFIG_STRUCT::MR<scomplex> { static constexpr idx_type def = BLIS_DEFAULT_MR_C, extent = BLIS_PACKDIM_MR_C; };
template <> struct TBLIS_CONFIG_STRUCT::MR<dcomplex> { static constexpr idx_type def = BLIS_DEFAULT_MR_Z, extent = BLIS_PACKDIM_MR_Z; };

template <> struct TBLIS_CONFIG_STRUCT::NR<   float> { static constexpr idx_type def = BLIS_DEFAULT_NR_S, extent = BLIS_PACKDIM_NR_S; };
template <> struct TBLIS_CONFIG_STRUCT::NR<  double> { static constexpr idx_type def = BLIS_DEFAULT_NR_D, extent = BLIS_PACKDIM_NR_D; };
template <> struct TBLIS_CONFIG_STRUCT::NR<scomplex> { static constexpr idx_type def = BLIS_DEFAULT_NR_C, extent = BLIS_PACKDIM_NR_C; };
template <> struct TBLIS_CONFIG_STRUCT::NR<dcomplex> { static constexpr idx_type def = BLIS_DEFAULT_NR_Z, extent = BLIS_PACKDIM_NR_Z; };

template <> struct TBLIS_CONFIG_STRUCT::KR<   float> { static constexpr idx_type def = BLIS_DEFAULT_KR_S, extent = BLIS_PACKDIM_KR_S; };
template <> struct TBLIS_CONFIG_STRUCT::KR<  double> { static constexpr idx_type def = BLIS_DEFAULT_KR_D, extent = BLIS_PACKDIM_KR_D; };
template <> struct TBLIS_CONFIG_STRUCT::KR<scomplex> { static constexpr idx_type def = BLIS_DEFAULT_KR_C, extent = BLIS_PACKDIM_KR_C; };
template <> struct TBLIS_CONFIG_STRUCT::KR<dcomplex> { static constexpr idx_type def = BLIS_DEFAULT_KR_Z, extent = BLIS_PACKDIM_KR_Z; };

template <> struct TBLIS_CONFIG_STRUCT::gemm_ukr<   float> { static constexpr gemm_ukr_t<   float> value = BLIS_SGEMM_UKERNEL; };
template <> struct TBLIS_CONFIG_STRUCT::gemm_ukr<  double> { static constexpr gemm_ukr_t<  double> value = BLIS_DGEMM_UKERNEL; };
template <> struct TBLIS_CONFIG_STRUCT::gemm_ukr<scomplex> { static constexpr gemm_ukr_t<scomplex> value = BLIS_CGEMM_UKERNEL; };
template <> struct TBLIS_CONFIG_STRUCT::gemm_ukr<dcomplex> { static constexpr gemm_ukr_t<dcomplex> value = BLIS_ZGEMM_UKERNEL; };

template <> struct TBLIS_CONFIG_STRUCT::gemm_row_major<   float> { static constexpr bool value = BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS; };
template <> struct TBLIS_CONFIG_STRUCT::gemm_row_major<  double> { static constexpr bool value = BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS; };
template <> struct TBLIS_CONFIG_STRUCT::gemm_row_major<scomplex> { static constexpr bool value = BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS; };
template <> struct TBLIS_CONFIG_STRUCT::gemm_row_major<dcomplex> { static constexpr bool value = BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS; };

#undef TBLIS_CONFIG_STRUCT
#undef TBLIS_CONFIG

}

#undef BLIS_KERNEL_H

#ifdef BLIS_TREE_BARRIER
#undef BLIS_TREE_BARRIER
#undef BLIS_TREE_BARRIER_ARITY
#endif

#undef BLIS_KERNEL_MACRO_DEFS_H


// -- MEMORY ALLOCATION --------------------------------------------------------

// Size of a virtual memory page. This is used to align certain memory
// buffers which are allocated and used internally.
#ifdef BLIS_PAGE_SIZE
#undef BLIS_PAGE_SIZE
#endif

// Number of named SIMD vector registers available for use.
#ifdef BLIS_SIMD_NUM_REGISTERS
#undef BLIS_SIMD_NUM_REGISTERS
#endif

// Size (in bytes) of each SIMD vector.
#ifdef BLIS_SIMD_SIZE
#undef BLIS_SIMD_SIZE
#endif

// Alignment size (in bytes) needed by the instruction set for aligned
// SIMD/vector instructions.
#ifdef BLIS_SIMD_ALIGN_SIZE
#undef BLIS_SIMD_ALIGN_SIZE
#endif

// The maximum size in bytes of local stack buffers within macro-kernel
// functions. These buffers are usually used to store a temporary copy
// of a single microtile. The reason we multiply by 2 is to handle induced
// methods, where we use real domain register blocksizes in units of
// complex elements. Specifically, the macro-kernels will need this larger
// micro-tile footprint, even though the virtual micro-kernels will only
// ever be writing to half (real or imaginary part) at a time.
#ifdef BLIS_STACK_BUF_MAX_SIZE
#undef BLIS_STACK_BUF_MAX_SIZE
#endif

// Alignment size used to align local stack buffers within macro-kernel
// functions.
#undef BLIS_STACK_BUF_ALIGN_SIZE

// Alignment size used when allocating memory dynamically from the operating
// system (eg: posix_memalign()). To disable heap alignment and just use
// malloc() instead, set this to 1.
#undef BLIS_HEAP_ADDR_ALIGN_SIZE

// Alignment size used when sizing leading dimensions of dynamically
// allocated memory.
#undef BLIS_HEAP_STRIDE_ALIGN_SIZE

// Alignment size used when allocating blocks to the internal memory
// pool (for packing buffers).
#undef BLIS_POOL_ADDR_ALIGN_SIZE


// -- Define row access bools --------------------------------------------------

// In this section we consider each datatype-specific "prefers contiguous rows"
// macro. If it is defined, we re-define it to be 1 (TRUE); otherwise, we
// define it to be 0 (FALSE).

// gemm micro-kernels

#undef BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS

#undef BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS

#undef BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS

#undef BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS


// -- Define default kernel names ----------------------------------------------

// In this section we consider each datatype-specific micro-kernel macro;
// if it is undefined, we define it to be the corresponding reference kernel.
// In the case of complex gemm micro-kernels, we also define special macros
// so that later on we can tell whether or not to employ the induced
// implementations. Note that in order to properly determine whether the
// induced method is a viable option, we need to be able to test the
// existence of the real gemm micro-kernels, which means we must consider
// the complex gemm micro-kernel cases *BEFORE* the real cases.

//
// Level-3
//

// gemm micro-kernels

#ifdef BLIS_ENABLE_INDUCED_SCOMPLEX
#undef BLIS_ENABLE_INDUCED_SCOMPLEX
#endif

#ifdef BLIS_ENABLE_INDUCED_DCOMPLEX
#undef BLIS_ENABLE_INDUCED_DCOMPLEX
#endif

#undef BLIS_SGEMM_UKERNEL
#undef BLIS_DGEMM_UKERNEL
#undef BLIS_CGEMM_UKERNEL
#undef BLIS_ZGEMM_UKERNEL

// gemmtrsm_l micro-kernels

#undef BLIS_SGEMMTRSM_L_UKERNEL
#undef BLIS_DGEMMTRSM_L_UKERNEL
#undef BLIS_CGEMMTRSM_L_UKERNEL
#undef BLIS_ZGEMMTRSM_L_UKERNEL

// gemmtrsm_u micro-kernels

#undef BLIS_SGEMMTRSM_U_UKERNEL
#undef BLIS_DGEMMTRSM_U_UKERNEL
#undef BLIS_CGEMMTRSM_U_UKERNEL
#undef BLIS_ZGEMMTRSM_U_UKERNEL

// trsm_l micro-kernels

#undef BLIS_STRSM_L_UKERNEL
#undef BLIS_DTRSM_L_UKERNEL
#undef BLIS_CTRSM_L_UKERNEL
#undef BLIS_ZTRSM_L_UKERNEL

// trsm_u micro-kernels

#undef BLIS_STRSM_U_UKERNEL
#undef BLIS_DTRSM_U_UKERNEL
#undef BLIS_CTRSM_U_UKERNEL
#undef BLIS_ZTRSM_U_UKERNEL

//
// Level-1m
//

// packm_2xk kernels

#undef BLIS_SPACKM_2XK_KERNEL
#undef BLIS_DPACKM_2XK_KERNEL
#undef BLIS_CPACKM_2XK_KERNEL
#undef BLIS_ZPACKM_2XK_KERNEL

// packm_3xk kernels

#undef BLIS_SPACKM_3XK_KERNEL
#undef BLIS_DPACKM_3XK_KERNEL
#undef BLIS_CPACKM_3XK_KERNEL
#undef BLIS_ZPACKM_3XK_KERNEL

// packm_4xk kernels

#undef BLIS_SPACKM_4XK_KERNEL
#undef BLIS_DPACKM_4XK_KERNEL
#undef BLIS_CPACKM_4XK_KERNEL
#undef BLIS_ZPACKM_4XK_KERNEL

// packm_6xk kernels

#undef BLIS_SPACKM_6XK_KERNEL
#undef BLIS_DPACKM_6XK_KERNEL
#undef BLIS_CPACKM_6XK_KERNEL
#undef BLIS_ZPACKM_6XK_KERNEL

// packm_8xk kernels

#undef BLIS_SPACKM_8XK_KERNEL
#undef BLIS_DPACKM_8XK_KERNEL
#undef BLIS_CPACKM_8XK_KERNEL
#undef BLIS_ZPACKM_8XK_KERNEL

// packm_10xk kernels

#undef BLIS_SPACKM_10XK_KERNEL
#undef BLIS_DPACKM_10XK_KERNEL
#undef BLIS_CPACKM_10XK_KERNEL
#undef BLIS_ZPACKM_10XK_KERNEL

// packm_12xk kernels

#undef BLIS_SPACKM_12XK_KERNEL
#undef BLIS_DPACKM_12XK_KERNEL
#undef BLIS_CPACKM_12XK_KERNEL
#undef BLIS_ZPACKM_12XK_KERNEL

// packm_14xk kernels

#undef BLIS_SPACKM_14XK_KERNEL
#undef BLIS_DPACKM_14XK_KERNEL
#undef BLIS_CPACKM_14XK_KERNEL
#undef BLIS_ZPACKM_14XK_KERNEL

// packm_16xk kernels

#undef BLIS_SPACKM_16XK_KERNEL
#undef BLIS_DPACKM_16XK_KERNEL
#undef BLIS_CPACKM_16XK_KERNEL
#undef BLIS_ZPACKM_16XK_KERNEL

// packm_30xk kernels

#undef BLIS_SPACKM_30XK_KERNEL
#undef BLIS_DPACKM_30XK_KERNEL
#undef BLIS_CPACKM_30XK_KERNEL
#undef BLIS_ZPACKM_30XK_KERNEL

// unpackm_2xk kernels

#undef BLIS_SUNPACKM_2XK_KERNEL
#undef BLIS_DUNPACKM_2XK_KERNEL
#undef BLIS_CUNPACKM_2XK_KERNEL
#undef BLIS_ZUNPACKM_2XK_KERNEL

// unpackm_4xk kernels

#undef BLIS_SUNPACKM_4XK_KERNEL
#undef BLIS_DUNPACKM_4XK_KERNEL
#undef BLIS_CUNPACKM_4XK_KERNEL
#undef BLIS_ZUNPACKM_4XK_KERNEL

// unpackm_6xk kernels

#undef BLIS_SUNPACKM_6XK_KERNEL
#undef BLIS_DUNPACKM_6XK_KERNEL
#undef BLIS_CUNPACKM_6XK_KERNEL
#undef BLIS_ZUNPACKM_6XK_KERNEL

// unpackm_8xk kernels

#undef BLIS_SUNPACKM_8XK_KERNEL
#undef BLIS_DUNPACKM_8XK_KERNEL
#undef BLIS_CUNPACKM_8XK_KERNEL
#undef BLIS_ZUNPACKM_8XK_KERNEL

// unpackm_10xk kernels

#undef BLIS_SUNPACKM_10XK_KERNEL
#undef BLIS_DUNPACKM_10XK_KERNEL
#undef BLIS_CUNPACKM_10XK_KERNEL
#undef BLIS_ZUNPACKM_10XK_KERNEL

// unpackm_12xk kernels

#undef BLIS_SUNPACKM_12XK_KERNEL
#undef BLIS_DUNPACKM_12XK_KERNEL
#undef BLIS_CUNPACKM_12XK_KERNEL
#undef BLIS_ZUNPACKM_12XK_KERNEL

// unpackm_14xk kernels

#undef BLIS_SUNPACKM_14XK_KERNEL
#undef BLIS_DUNPACKM_14XK_KERNEL
#undef BLIS_CUNPACKM_14XK_KERNEL
#undef BLIS_ZUNPACKM_14XK_KERNEL

// unpackm_16xk kernels

#undef BLIS_SUNPACKM_16XK_KERNEL
#undef BLIS_DUNPACKM_16XK_KERNEL
#undef BLIS_CUNPACKM_16XK_KERNEL
#undef BLIS_ZUNPACKM_16XK_KERNEL

//
// Level-1f
//

// axpy2v kernels

//#ifndef       AXPY2V_KERNEL
//#undef
//#endif

#undef BLIS_SAXPY2V_KERNEL
#undef BLIS_DAXPY2V_KERNEL
#undef BLIS_CAXPY2V_KERNEL
#undef BLIS_ZAXPY2V_KERNEL

// dotaxpyv kernels

//#ifndef       DOTAXPYV_KERNEL
//#undef
//#endif

#undef BLIS_SDOTAXPYV_KERNEL
#undef BLIS_DDOTAXPYV_KERNEL
#undef BLIS_CDOTAXPYV_KERNEL
#undef BLIS_ZDOTAXPYV_KERNEL

// axpyf kernels

//#ifndef       AXPYF_KERNEL
//#undef
//#endif

#undef BLIS_SAXPYF_KERNEL
#undef BLIS_DAXPYF_KERNEL
#undef BLIS_CAXPYF_KERNEL
#undef BLIS_ZAXPYF_KERNEL

// dotxf kernels

//#ifndef       DOTXF_KERNEL
//#undef
//#endif

#undef BLIS_SDOTXF_KERNEL
#undef BLIS_DDOTXF_KERNEL
#undef BLIS_CDOTXF_KERNEL
#undef BLIS_ZDOTXF_KERNEL

// dotxaxpyf kernels

//#ifndef       DOTXAXPYF_KERNEL
//#undef
//#endif

#undef BLIS_SDOTXAXPYF_KERNEL
#undef BLIS_DDOTXAXPYF_KERNEL
#undef BLIS_CDOTXAXPYF_KERNEL
#undef BLIS_ZDOTXAXPYF_KERNEL

//
// Level-1v
//

// addv kernels

//#ifndef       ADDV_KERNEL
//#undef
//#endif

#undef BLIS_SADDV_KERNEL
#undef BLIS_DADDV_KERNEL
#undef BLIS_CADDV_KERNEL
#undef BLIS_ZADDV_KERNEL

// axpyv kernels

//#ifndef       AXPYV_KERNEL
//#undef
//#endif

#undef BLIS_SAXPYV_KERNEL
#undef BLIS_DAXPYV_KERNEL
#undef BLIS_CAXPYV_KERNEL
#undef BLIS_ZAXPYV_KERNEL

// copyv kernels

//#ifndef       COPYV_KERNEL
//#undef
//#endif

#undef BLIS_SCOPYV_KERNEL
#undef BLIS_DCOPYV_KERNEL
#undef BLIS_CCOPYV_KERNEL
#undef BLIS_ZCOPYV_KERNEL

// dotv kernels

//#ifndef       DOTV_KERNEL
//#undef
//#endif

#undef BLIS_SDOTV_KERNEL
#undef BLIS_DDOTV_KERNEL
#undef BLIS_CDOTV_KERNEL
#undef BLIS_ZDOTV_KERNEL

// dotxv kernels

//#ifndef       DOTXV_KERNEL
//#undef
//#endif

#undef BLIS_SDOTXV_KERNEL
#undef BLIS_DDOTXV_KERNEL
#undef BLIS_CDOTXV_KERNEL
#undef BLIS_ZDOTXV_KERNEL

// invertv kernels

//#ifndef       INVERTV_KERNEL
//#undef
//#endif

#undef BLIS_SINVERTV_KERNEL
#undef BLIS_DINVERTV_KERNEL
#undef BLIS_CINVERTV_KERNEL
#undef BLIS_ZINVERTV_KERNEL

// scal2v kernels

//#ifndef       SCAL2V_KERNEL
//#undef
//#endif

#undef BLIS_SSCAL2V_KERNEL
#undef BLIS_DSCAL2V_KERNEL
#undef BLIS_CSCAL2V_KERNEL
#undef BLIS_ZSCAL2V_KERNEL

// scalv kernels

//#ifndef       SCALV_KERNEL
//#undef
//#endif

#undef BLIS_SSCALV_KERNEL
#undef BLIS_DSCALV_KERNEL
#undef BLIS_CSCALV_KERNEL
#undef BLIS_ZSCALV_KERNEL

// setv kernels

//#ifndef       SETV_KERNEL
//#undef
//#endif

#undef BLIS_SSETV_KERNEL
#undef BLIS_DSETV_KERNEL
#undef BLIS_CSETV_KERNEL
#undef BLIS_ZSETV_KERNEL

// subv kernels

//#ifndef       SUBV_KERNEL
//#undef
//#endif

#undef BLIS_SSUBV_KERNEL
#undef BLIS_DSUBV_KERNEL
#undef BLIS_CSUBV_KERNEL
#undef BLIS_ZSUBV_KERNEL

// swapv kernels

//#ifndef       SWAPV_KERNEL
//#undef
//#endif

#undef BLIS_SSWAPV_KERNEL
#undef BLIS_DSWAPV_KERNEL
#undef BLIS_CSWAPV_KERNEL
#undef BLIS_ZSWAPV_KERNEL


// -- Define default blocksize macros ------------------------------------------

//
// Define level-3 cache blocksizes.
//

// Define MC minimum

#undef BLIS_DEFAULT_MC_S
#undef BLIS_DEFAULT_MC_D
#undef BLIS_DEFAULT_MC_C
#undef BLIS_DEFAULT_MC_Z

// Define KC minimum

#undef BLIS_DEFAULT_KC_S
#undef BLIS_DEFAULT_KC_D
#undef BLIS_DEFAULT_KC_C
#undef BLIS_DEFAULT_KC_Z

// Define NC minimum

#undef BLIS_DEFAULT_NC_S
#undef BLIS_DEFAULT_NC_D
#undef BLIS_DEFAULT_NC_C
#undef BLIS_DEFAULT_NC_Z

// Define MC maximum

#undef BLIS_MAXIMUM_MC_S
#undef BLIS_MAXIMUM_MC_D
#undef BLIS_MAXIMUM_MC_C
#undef BLIS_MAXIMUM_MC_Z

// Define KC maximum

#undef BLIS_MAXIMUM_KC_S
#undef BLIS_MAXIMUM_KC_D
#undef BLIS_MAXIMUM_KC_C
#undef BLIS_MAXIMUM_KC_Z

// Define NC maximum

#undef BLIS_MAXIMUM_NC_S
#undef BLIS_MAXIMUM_NC_D
#undef BLIS_MAXIMUM_NC_C
#undef BLIS_MAXIMUM_NC_Z

//
// Define level-3 register blocksizes.
//

// Define MR

#undef BLIS_DEFAULT_MR_S
#undef BLIS_DEFAULT_MR_D
#undef BLIS_DEFAULT_MR_C
#undef BLIS_DEFAULT_MR_Z

// Define NR

#undef BLIS_DEFAULT_NR_S
#undef BLIS_DEFAULT_NR_D
#undef BLIS_DEFAULT_NR_C
#undef BLIS_DEFAULT_NR_Z

// Define KR

#undef BLIS_DEFAULT_KR_S
#undef BLIS_DEFAULT_KR_D
#undef BLIS_DEFAULT_KR_C
#undef BLIS_DEFAULT_KR_Z

// Define MR packdim

#undef BLIS_PACKDIM_MR_S
#undef BLIS_PACKDIM_MR_D
#undef BLIS_PACKDIM_MR_C
#undef BLIS_PACKDIM_MR_Z

// Define NR packdim

#undef BLIS_PACKDIM_NR_S
#undef BLIS_PACKDIM_NR_D
#undef BLIS_PACKDIM_NR_C
#undef BLIS_PACKDIM_NR_Z

// Define KR packdim

#undef BLIS_PACKDIM_KR_S
#undef BLIS_PACKDIM_KR_D
#undef BLIS_PACKDIM_KR_C
#undef BLIS_PACKDIM_KR_Z

//
// Define level-2 blocksizes.
//

// NOTE: These values determine high-level cache blocking for level-2
// operations ONLY. So, if gemv is performed with a 2000x2000 matrix A and
// MC = NC = 1000, then a total of four unblocked (or unblocked fused)
// gemv subproblems are called. The blocked algorithms are only useful in
// that they provide the opportunity for packing vectors. (Matrices can also
// be packed here, but this tends to be much too expensive in practice to
// actually employ.)

#undef BLIS_DEFAULT_L2_MC_S
#undef BLIS_DEFAULT_L2_NC_S
#undef BLIS_DEFAULT_L2_MC_D
#undef BLIS_DEFAULT_L2_NC_D

#undef BLIS_DEFAULT_L2_MC_C
#undef BLIS_DEFAULT_L2_NC_C
#undef BLIS_DEFAULT_L2_MC_Z
#undef BLIS_DEFAULT_L2_NC_Z

//
// Define level-1f fusing factors.
//

// Global level-1f fusing factors.

#undef BLIS_L1F_FUSE_FAC_S
#undef BLIS_L1F_FUSE_FAC_D
#undef BLIS_L1F_FUSE_FAC_C
#undef BLIS_L1F_FUSE_FAC_Z

// axpyf

#undef BLIS_AXPYF_FUSE_FAC_S
#undef BLIS_AXPYF_FUSE_FAC_D
#undef BLIS_AXPYF_FUSE_FAC_C
#undef BLIS_AXPYF_FUSE_FAC_Z

// dotxf

#undef BLIS_DOTXF_FUSE_FAC_S
#undef BLIS_DOTXF_FUSE_FAC_D
#undef BLIS_DOTXF_FUSE_FAC_C
#undef BLIS_DOTXF_FUSE_FAC_Z

// dotxaxpyf

#undef BLIS_DOTXAXPYF_FUSE_FAC_S
#undef BLIS_DOTXAXPYF_FUSE_FAC_D
#undef BLIS_DOTXAXPYF_FUSE_FAC_C
#undef BLIS_DOTXAXPYF_FUSE_FAC_Z

//
// Define level-1v blocksizes.
//

// NOTE: Register blocksizes for vectors are used when packing
// non-contiguous vectors. Similar to that of KR, they can
// typically be set to 1.

#undef BLIS_DEFAULT_VR_S
#undef BLIS_DEFAULT_VR_D
#undef BLIS_DEFAULT_VR_C
#undef BLIS_DEFAULT_VR_Z


// -- Abbreiviated kernel blocksize macros -------------------------------------

// Here, we shorten the blocksizes defined in bli_kernel.h so that they can
// derived via the PASTEMAC macro.

// Default (minimum) cache blocksizes

#undef bli_smc
#undef bli_skc
#undef bli_snc

#undef bli_dmc
#undef bli_dkc
#undef bli_dnc

#undef bli_cmc
#undef bli_ckc
#undef bli_cnc

#undef bli_zmc
#undef bli_zkc
#undef bli_znc

// Register blocksizes

#undef bli_smr
#undef bli_skr
#undef bli_snr

#undef bli_dmr
#undef bli_dkr
#undef bli_dnr

#undef bli_cmr
#undef bli_ckr
#undef bli_cnr

#undef bli_zmr
#undef bli_zkr
#undef bli_znr

// Extended (maximum) cache blocksizes

#undef bli_smaxmc
#undef bli_smaxkc
#undef bli_smaxnc

#undef bli_dmaxmc
#undef bli_dmaxkc
#undef bli_dmaxnc

#undef bli_cmaxmc
#undef bli_cmaxkc
#undef bli_cmaxnc

#undef bli_zmaxmc
#undef bli_zmaxkc
#undef bli_zmaxnc

// Extended (packing) register blocksizes

#undef bli_spackmr
#undef bli_spackkr
#undef bli_spacknr

#undef bli_dpackmr
#undef bli_dpackkr
#undef bli_dpacknr

#undef bli_cpackmr
#undef bli_cpackkr
#undef bli_cpacknr

#undef bli_zpackmr
#undef bli_zpackkr
#undef bli_zpacknr

// Level-1f fusing factors

#undef bli_saxpyf_fusefac
#undef bli_daxpyf_fusefac
#undef bli_caxpyf_fusefac
#undef bli_zaxpyf_fusefac

#undef bli_sdotxf_fusefac
#undef bli_ddotxf_fusefac
#undef bli_cdotxf_fusefac
#undef bli_zdotxf_fusefac

#undef bli_sdotxaxpyf_fusefac
#undef bli_ddotxaxpyf_fusefac
#undef bli_cdotxaxpyf_fusefac
#undef bli_zdotxaxpyf_fusefac
