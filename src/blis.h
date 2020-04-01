#ifndef _TBLIS_BLIS_H_
#define _TBLIS_BLIS_H_

#include <stdint.h>

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
);

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
