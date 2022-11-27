#ifndef TBLIS_KERNELS_3M_GEMM_HPP
#define TBLIS_KERNELS_3M_GEMM_HPP 1

#include <tblis/internal/types.hpp>
#include <tblis/blis.h>

namespace tblis
{

#define EXTERN_UPDATE_NN_UKR(name) \
extern void name(tblis::stride_type m, \
                 tblis::stride_type n, \
                 const void* ab, \
                 const void* d, tblis::stride_type inc_d, \
                 const void* e, tblis::stride_type inc_e, \
                 const void* beta, \
                 void* c, tblis::stride_type rs_c, \
                          tblis::stride_type cs_c);

#define EXTERN_UPDATE_SS_UKR(name) \
extern void name(tblis::stride_type m, \
                 tblis::stride_type n, \
                 const void* ab, \
                 const void* beta, \
                 void* c, const tblis::stride_type* rscat_c, \
                          const tblis::stride_type* cscat_c);

#define EXTERN_GEMM_UKR(name) \
extern void name(tblis::stride_type m, \
                 tblis::stride_type n, \
                 tblis::stride_type k, \
                 const void* a, const void* b, \
                 const void* beta, \
                 void* c, tblis::stride_type rs_c, \
                          tblis::stride_type cs_c, \
                          auxinfo_t* aux);

template <typename Config, typename T>
void update_nn_ukr_def(stride_type m, stride_type n,
                       const void* p_ab_,
                       const void* p_d_, stride_type inc_d,
                       const void* p_e_, stride_type inc_e,
                       const void* beta_,
                       void* p_c_, stride_type rs_c, stride_type cs_c)
{
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;
    constexpr bool row_major = Config::template gemm_row_major<T>::value;

    T beta = *static_cast<const T*>(beta_);

    const T* TBLIS_RESTRICT p_ab = static_cast<const T*>(p_ab_);
    const T* TBLIS_RESTRICT p_d  = static_cast<const T*>(p_d_);
    const T* TBLIS_RESTRICT p_e  = static_cast<const T*>(p_e_);
          T* TBLIS_RESTRICT p_c  = static_cast<      T*>(p_c_);

    if (row_major)
    {
        if (beta == T(0))
        {
            if (p_d && p_e)
            {
                if (m == MR && n == NR && cs_c == 1 && inc_e == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] =
                                p_ab[i*NR + j] * p_d[i*inc_d] * p_e[j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i*NR + j] * p_d[i*inc_d] * p_e[j*inc_e];
                }
            }
            else if (p_d)
            {
                if (m == MR && n == NR && cs_c == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] = p_ab[i*NR + j] * p_d[i*inc_d];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] = p_ab[i*NR + j] * p_d[i*inc_d];
                }
            }
            else if (p_e)
            {
                if (m == MR && n == NR && cs_c == 1 && inc_e == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] = p_ab[i*NR + j] * p_e[j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] = p_ab[i*NR + j] * p_e[j*inc_e];
                }
            }
            else
            {
                if (m == MR && n == NR && cs_c == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] = p_ab[i*NR + j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] = p_ab[i*NR + j];
                }
            }
        }
        else
        {
            if (p_d && p_e)
            {
                if (m == MR && n == NR && cs_c == 1 && inc_e == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] =
                                p_ab[i*NR + j] * p_d[i*inc_d] * p_e[j] +
                                beta*p_c[i*rs_c + j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i*NR + j] * p_d[i*inc_d] * p_e[j*inc_e] +
                                beta*p_c[i*rs_c + j*cs_c];
                }
            }
            else if (p_d)
            {
                if (m == MR && n == NR && cs_c == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] =
                                p_ab[i*NR + j] * p_d[i*inc_d] +
                                beta*p_c[i*rs_c + j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i*NR + j] * p_d[i*inc_d] +
                                beta*p_c[i*rs_c + j*cs_c];
                }
            }
            else if (p_e)
            {
                if (m == MR && n == NR && cs_c == 1 && inc_e == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] =
                                p_ab[i*NR + j] * p_e[j] +
                                beta*p_c[i*rs_c + j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i*NR + j] * p_e[j*inc_e] +
                                beta*p_c[i*rs_c + j*cs_c];
                }
            }
            else
            {
                if (m == MR && n == NR && cs_c == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        #pragma omp simd
                        for (len_type j = 0;j < NR;j++)
                            p_c[i*rs_c + j] =
                                p_ab[i*NR + j] + beta*p_c[i*rs_c + j];
                }
                else
                {
                    for (len_type i = 0;i < m;i++)
                        for (len_type j = 0;j < n;j++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i*NR + j] + beta*p_c[i*rs_c + j*cs_c];
                }
            }
        }
    }
    else
    {
        if (beta == T(0))
        {
            if (p_d && p_e)
            {
                if (m == MR && n == NR && rs_c == 1 && inc_d == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] =
                                p_ab[i + j*MR] * p_d[i] * p_e[j*inc_e];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i + j*MR] * p_d[i*inc_d] * p_e[j*inc_e];
                }
            }
            else if (p_d)
            {
                if (m == MR && n == NR && rs_c == 1 && inc_d == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] = p_ab[i + j*MR] * p_d[i];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR] * p_d[i*inc_d];
                }
            }
            else if (p_e)
            {
                if (m == MR && n == NR && rs_c == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] = p_ab[i + j*MR] * p_e[j*inc_e];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR] * p_e[j*inc_e];
                }
            }
            else
            {
                if (m == MR && n == NR && rs_c == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] = p_ab[i + j*MR];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR];
                }
            }
        }
        else
        {
            if (p_d && p_e)
            {
                if (m == MR && n == NR && rs_c == 1 && inc_d == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] =
                                p_ab[i + j*MR] * p_d[i] * p_e[j*inc_e] +
                                beta*p_c[i + j*cs_c];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i + j*MR] * p_d[i*inc_d] * p_e[j*inc_e] +
                                beta*p_c[i*rs_c + j*cs_c];
                }
            }
            else if (p_d)
            {
                if (m == MR && n == NR && rs_c == 1 && inc_d == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] =
                                p_ab[i + j*MR] * p_d[i] +
                                beta*p_c[i + j*cs_c];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i + j*MR] * p_d[i*inc_d] +
                                beta*p_c[i*rs_c + j*cs_c];
                }
            }
            else if (p_e)
            {
                if (m == MR && n == NR && rs_c == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] =
                                p_ab[i + j*MR] * p_e[j*inc_e] +
                                beta*p_c[i + j*cs_c];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i + j*MR] * p_e[j*inc_e] +
                                beta*p_c[i*rs_c + j*cs_c];
                }
            }
            else
            {
                if (m == MR && n == NR && rs_c == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        #pragma omp simd
                        for (len_type i = 0;i < MR;i++)
                            p_c[i + j*cs_c] =
                                p_ab[i + j*MR] + beta*p_c[i + j*cs_c];
                }
                else
                {
                    for (len_type j = 0;j < n;j++)
                        for (len_type i = 0;i < m;i++)
                            p_c[i*rs_c + j*cs_c] =
                                p_ab[i + j*MR] + beta*p_c[i*rs_c + j*cs_c];
                }
            }
        }
    }
}

template <typename Config, typename T>
void update_ss_ukr_def(stride_type m, stride_type n,
                       const void* p_ab_,
                       const void* beta_,
                       void* p_c_, const stride_type* rscat_c_,
                                   const stride_type* cscat_c_)
{
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;
    constexpr bool row_major = Config::template gemm_row_major<T>::value;

    T beta = *static_cast<const T*>(beta_);

    const T* TBLIS_RESTRICT p_ab = static_cast<const T*>(p_ab_);
          T* TBLIS_RESTRICT p_c  = static_cast<      T*>(p_c_);

    const stride_type* TBLIS_RESTRICT rscat_c = rscat_c_;
    const stride_type* TBLIS_RESTRICT cscat_c = cscat_c_;

    if (row_major)
    {
        if (beta == T(0))
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    p_c[rscat_c[i] + cscat_c[j]] = p_ab[i*NR + j];
        }
        else
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    p_c[rscat_c[i] + cscat_c[j]] =
                        p_ab[i*NR + j] + beta*p_c[rscat_c[i] + cscat_c[j]];
        }
    }
    else
    {
        if (beta == T(0))
        {
            for (len_type j = 0;j < n;j++)
                for (len_type i = 0;i < m;i++)
                    p_c[rscat_c[i] + cscat_c[j]] = p_ab[i + j*MR];
        }
        else
        {
            for (len_type j = 0;j < n;j++)
                for (len_type i = 0;i < m;i++)
                    p_c[rscat_c[i] + cscat_c[j]] =
                        p_ab[i + j*MR] + beta*p_c[rscat_c[i] + cscat_c[j]];
        }
    }
}

template <typename Config, typename T>
void gemm_ukr_def(stride_type m, stride_type n, stride_type k,
                  const void* p_a_, const void* p_b_,
                  const void* beta_,
                  void* p_c_, stride_type rs_c, stride_type cs_c,
                  auxinfo_t*)
{
    constexpr len_type MR = Config::template gemm_mr<T>::def;
    constexpr len_type NR = Config::template gemm_nr<T>::def;

    T beta = *static_cast<const T*>(beta_);

    const T* TBLIS_RESTRICT p_a = static_cast<const T*>(p_a_);
    const T* TBLIS_RESTRICT p_b = static_cast<const T*>(p_b_);
          T* TBLIS_RESTRICT p_c = static_cast<      T*>(p_c_);

    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int i = 0;i < MR;i++)
            #pragma omp simd
            for (int j = 0;j < NR;j++)
                p_ab[i*NR + j] += p_a[i] * p_b[j];

        p_a += MR;
        p_b += NR;
    }

    if (m == MR && n == NR && cs_c == 1)
    {
        if (beta == T(0))
        {
            for (len_type i = 0;i < MR;i++)
                #pragma omp simd
                for (len_type j = 0;j < NR;j++)
                    p_c[i*rs_c + j] = p_ab[i*NR + j];
        }
        else
        {
            for (len_type i = 0;i < MR;i++)
                #pragma omp simd
                for (len_type j = 0;j < NR;j++)
                    p_c[i*rs_c + j] = p_ab[i*NR + j] + beta*p_c[i*rs_c + j];
        }
    }
    else
    {
        if (beta == T(0))
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    p_c[i*rs_c + j*cs_c] = p_ab[i*NR + j];
        }
        else
        {
            for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    p_c[i*rs_c + j*cs_c] = p_ab[i*NR + j] +
                        beta*p_c[i*rs_c + j*cs_c];
        }
    }
}

}

#endif //TBLIS_KERNELS_3M_GEMM_HPP
