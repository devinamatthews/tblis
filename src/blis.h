#ifndef _TBLIS_BLIS_H_
#define _TBLIS_BLIS_H_

#include <stdint.h>
#include <stdbool.h>

#include "tblis_config.h"
#include "util/basic_types.h"

#ifdef __cplusplus
using namespace tblis;
#endif

typedef len_type dim_t;
typedef stride_type inc_t;

typedef struct { void *a_next, *b_next, *c_prefetch; } auxinfo_t;
typedef struct {} cntx_t;

#define bli_auxinfo_next_a(x) x->a_next;
#define bli_auxinfo_next_b(x) x->b_next;

typedef enum
{
    BLIS_NO_CONJUGATE      = 0x0,
    BLIS_CONJUGATE         = (1<<4)
} conj_t;

#define BLIS_STACK_BUF_MAX_SIZE 32*32*16

#define PASTEMAC2(x,y) bli_##x##y
#define PASTEMAC(x,y) PASTEMAC2(x,y)

#define bli_sctype float
#define bli_dctype double
#define bli_cctype scomplex
#define bli_zctype dcomplex

#define bli_sset0s(x) do { x = 0.0f; } while(0)
#define bli_dset0s(x) do { x = 0.0; } while(0)
#define bli_cset0s(x) do { x = 0.0f; } while(0)
#define bli_zset0s(x) do { x = 0.0; } while(0)

#define bli_xpbys_mxn(ch,m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b) \
do \
{ \
    PASTEMAC(ch,ctype) beta = *(beta_); \
    PASTEMAC(ch,ctype)* __restrict__ a = a_; \
    PASTEMAC(ch,ctype)* __restrict__ b = b_; \
\
    if (beta == 0.0) \
    { \
        for (dim_t i = 0;i < m;i++) \
        for (dim_t j = 0;j < n;j++) \
            b[i*(rs_b) + j*(cs_b)] = a[i*(rs_a) + j*(cs_a)]; \
    } \
    else \
    { \
        for (dim_t i = 0;i < m;i++) \
        for (dim_t j = 0;j < n;j++) \
            b[i*(rs_b) + j*(cs_b)] = beta*b[i*(rs_b) + j*(cs_b)] + \
                                          a[i*(rs_a) + j*(cs_a)]; \
    } \
} \
while (0)

#define bli_sxpbys_mxn(m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b) bli_xpbys_mxn(s,m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b)
#define bli_dxpbys_mxn(m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b) bli_xpbys_mxn(d,m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b)
#define bli_cxpbys_mxn(m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b) bli_xpbys_mxn(c,m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b)
#define bli_zxpbys_mxn(m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b) bli_xpbys_mxn(z,m,n,a_,rs_a,cs_a,beta_,b_,rs_b,cs_b)

//
// Macros for edge-case handling within gemm microkernels.
//

// -- Setup helper macros --

#define GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,alignment) \
\
	PASTEMAC(ch,ctype)* restrict _beta   = beta; \
	PASTEMAC(ch,ctype)* restrict _c      = c; \
	const inc_t                  _rs_c   = rs_c; \
	const inc_t                  _cs_c   = cs_c; \
	PASTEMAC(ch,ctype)           _ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( PASTEMAC(ch,ctype) ) ] \
	                                  __attribute__((aligned(alignment))); \
	const inc_t                  _rs_ct  = row_major ? nr :  1; \
	const inc_t                  _cs_ct  = row_major ?  1 : mr;

#define GEMM_UKR_SETUP_CT_POST(ch) \
\
	PASTEMAC(ch,ctype) _zero; \
	PASTEMAC(ch,set0s)( _zero ); \
	\
	if ( _use_ct ) \
	{ \
		c = _ct; \
		rs_c = _rs_ct; \
		cs_c = _cs_ct; \
		beta = &_zero; \
	}

// -- Setup macros --

#define GEMM_UKR_SETUP_CT(ch,mr,nr,row_major) \
\
	/* Scenario 1: the ukernel contains assembly-level support only for its
	   IO preference (e.g. only row-oriented or only column-oriented IO).
	   Use a temporary microtile for the other two cases as well as edge
	   cases. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_AMBI(ch,mr,nr,row_major) \
\
	/* Scenario 2: the ukernel contains assembly-level support for its IO
	   preference as well as its opposite via in-register transpose
	   (e.g. both row- and column-oriented IO). Use a temporary microtile
	   for the general stride case as well as edge cases. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( cs_c != 1 && rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_ANY(ch,mr,nr,row_major) \
\
	/* Scenario 3: Similar to (2) where the assembly region also supports
	   general stride I0. Use a temporary microtile only for edge cases. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( m != mr || n != nr ); \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_ALIGNED(ch,mr,nr,row_major,alignment) \
\
	/* Scenario 4: Similar to (1), but uses temporary microtile to handle
	   cases where the pointer to the C microtile is not aligned. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,alignment); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr || \
	                     ( (uintptr_t)_c % alignment ) || \
	                     ( ( ( row_major ? _rs_c : _cs_c )*sizeof( PASTEMAC(ch,ctype) ) ) % alignment ); \
	GEMM_UKR_SETUP_CT_POST(ch);

// -- Flush macros --

#define GEMM_UKR_FLUSH_CT(ch) \
\
	/* If we actually used the temporary microtile, accumulate it to the output
	   microtile. */ \
	if ( _use_ct ) \
	{ \
		PASTEMAC(ch,xpbys_mxn) \
		( \
		  m, n, \
		  _ct, _rs_ct, _cs_ct, \
		  _beta, \
		  _c,  _rs_c,  _cs_c \
		); \
	}

#define EXTERN_BLIS_GEMM_UKR(name) \
extern "C" void name \
( \
  dim_t      k, \
  void*      alpha, \
  void*      a, \
  void*      b, \
  void*      beta, \
  void*      c, inc_t rs_c, inc_t cs_c, \
  auxinfo_t* data, \
  cntx_t*    cntx \
)

#ifdef __cplusplus

using blis_gemm_ukr_t =
void (*)(dim_t      k,
         void*      alpha,
         void*      a,
         void*      b,
         void*      beta,
         void*      c, inc_t rs_c, inc_t cs_c,
         auxinfo_t* data,
         cntx_t*    cntx);

template <typename Config, typename T, blis_gemm_ukr_t UKR>
void blis_wrap_gemm_ukr(stride_type m, stride_type n, stride_type k,
                        const void* p_a, const void* p_b,
                        const void* beta,
                              void* p_c, stride_type rs_c, stride_type cs_c,
                        auxinfo_t* aux)
{
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    constexpr len_type RS = Config::template gemm_row_major<T>::value ? NR : 1;
    constexpr len_type CS = Config::template gemm_row_major<T>::value ? 1 : MR;

    T one = 1.0, zero = 0.0;
    T p_ab[MR*NR] __attribute__((aligned(64)));

    if (m == MR && n == NR)
    {
        UKR(k, &one, const_cast<void*>(p_a), const_cast<void*>(p_b),
            const_cast<void*>(beta), p_c, rs_c, cs_c, aux, nullptr);
    }
    else
    {
        UKR(k, &one, const_cast<void*>(p_a), const_cast<void*>(p_b),
            &zero, p_ab, RS, CS, aux, nullptr);

        Config::template update_nn_ukr<T>::value(m, n, p_ab, nullptr, 0, nullptr, 0,
                                                 beta, p_c, rs_c, cs_c);
    }
}

#endif

#endif
