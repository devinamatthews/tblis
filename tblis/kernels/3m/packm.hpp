#ifndef TBLIS_KERNELS_3M_PACKM_HPP
#define TBLIS_KERNELS_3M_PACKM_HPP 1

#include <tblis/internal/types.hpp>
#include <tblis/internal/alignment.hpp>

namespace tblis
{

#define EXTERN_PACK_NN_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const void* alpha, bool conj, \
                 const void* p_a, tblis::stride_type rs_a, \
                                  tblis::stride_type cs_a, \
                 const void* p_d, tblis::stride_type inc_d, \
                 const void* p_e, tblis::stride_type inc_e, \
                 void* p_ap)

#define EXTERN_PACK_SS_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const void* alpha, bool conj, \
                 const void* p_a, const tblis::stride_type* rscat_a, \
                                  const tblis::stride_type* cscat_a, \
                 void* p_ap)

#define EXTERN_PACK_NB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const void* alpha, bool conj, \
                 const void* p_a, tblis::stride_type rs_a, \
                                  const tblis::stride_type* cscat_a, \
                                  const tblis::stride_type* cbs_a, \
                 void* p_ap)

#define EXTERN_PACK_SS_SCAL_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const void* alpha, bool conj, \
                 const void* p_a, const tblis::stride_type* rscat_a, \
                                  const void* rscale_a, \
                                  const tblis::stride_type* cscat_a, \
                                  const void* cscale_a, \
                 void* p_ap)

template <typename Config, typename T, int Mat>
void pack_nn_ukr_def(len_type m, len_type k, const void* alpha_, bool conj,
                     const void* p_a_, stride_type rs_a, stride_type cs_a,
                     const void* p_d_, stride_type inc_d,
                     const void* p_e_, stride_type inc_e,
                     void* p_ap_)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;
    constexpr len_type KE = Config::template gemm_kr<T>::extent;

    T alpha = *static_cast<const T*>(alpha_);

    const T* TBLIS_RESTRICT p_a  = static_cast<const T*>(p_a_ );
          T* TBLIS_RESTRICT p_ap = static_cast<      T*>(p_ap_);
    const T* TBLIS_RESTRICT p_d  = static_cast<const T*>(p_d_ );
    const T* TBLIS_RESTRICT p_e  = static_cast<const T*>(p_e_ );

    if (conj && is_complex<T>::value)
    {
        if (p_d && p_e)
        {
            if (m == MR && rs_a == 1 && inc_d == 1 && inc_e == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * tblis::conj(p_a[mr + p*cs_a]) * p_d[mr] * p_e[p];
            }
            else if (m == MR && cs_a == 1 && inc_d == 1 && inc_e == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[rs_a*mr + p+kr]) * p_d[mr] * p_e[p+kr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + p]) * p_d[mr] * p_e[p];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + cs_a*p]) * p_d[inc_d*mr] * p_e[inc_e*p];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
        else if (p_d)
        {
            if (m == MR && rs_a == 1 && inc_d == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * tblis::conj(p_a[mr + p*cs_a]) * p_d[mr];
            }
            else if (m == MR && cs_a == 1 && inc_d == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[rs_a*mr + p+kr]) * p_d[mr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + p]) * p_d[mr];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + cs_a*p]) * p_d[inc_d*mr];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
        else if (p_e)
        {
            if (m == MR && rs_a == 1 && inc_e == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * tblis::conj(p_a[mr + p*cs_a]) * p_e[p];
            }
            else if (m == MR && cs_a == 1 && inc_e == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[rs_a*mr + p+kr]) * p_e[p+kr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + p]) * p_e[p];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + cs_a*p]) * p_e[inc_e*p];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
        else
        {
            if (m == MR && rs_a == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * tblis::conj(p_a[mr + p*cs_a]);
            }
            else if (m == MR && cs_a == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[rs_a*mr + p+kr]);

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + p]);
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + cs_a*p]);

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
    }
    else
    {
        if (p_d && p_e)
        {
            if (m == MR && rs_a == 1 && inc_d == 1 && inc_e == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * p_a[mr + p*cs_a] * p_d[mr] * p_e[p];
            }
            else if (m == MR && cs_a == 1 && inc_d == 1 && inc_e == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * p_a[rs_a*mr + p+kr] * p_d[mr] * p_e[p+kr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + p] * p_d[mr] * p_e[p];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + cs_a*p] * p_d[inc_d*mr] * p_e[inc_e*p];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
        else if (p_d)
        {
            if (m == MR && rs_a == 1 && inc_d == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * p_a[mr + p*cs_a] * p_d[mr];
            }
            else if (m == MR && cs_a == 1 && inc_d == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * p_a[rs_a*mr + p+kr] * p_d[mr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + p] * p_d[mr];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + cs_a*p] * p_d[inc_d*mr];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
        else if (p_e)
        {
            if (m == MR && rs_a == 1 && inc_e == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * p_a[mr + p*cs_a] * p_e[p];
            }
            else if (m == MR && cs_a == 1 && inc_e == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * p_a[rs_a*mr + p+kr] * p_e[p+kr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + p] * p_e[p];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + cs_a*p] * p_e[inc_e*p];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
        else
        {
            if (m == MR && rs_a == 1)
            {
                for (len_type p = 0;p < k;p++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + p*ME] = alpha * p_a[mr + p*cs_a];
            }
            else if (m == MR && cs_a == 1)
            {
                len_type p = 0;
                for (;p < k-KR;p += KR)
                for (len_type kr = 0;kr < KR;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*(p+kr)] = alpha * p_a[rs_a*mr + p+kr];

                for (;p < k;p++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + p];
            }
            else
            {
                for (len_type p = 0;p < k;p++)
                {
                    for (len_type mr = 0;mr < m;mr++)
                        p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + cs_a*p];

                    for (len_type mr = m;mr < MR;mr++)
                        p_ap[mr + ME*p] = T();
                }
            }
        }
    }

    for (len_type p = k;p < round_up(k,KE);p++)
        #pragma omp simd
        for (len_type mr = 0;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
}

template <typename Config, typename T, int Mat>
void pack_ss_ukr_def(len_type m, len_type k, const void* alpha_, bool conj,
                     const void* p_a_,
                     const stride_type* rscat_a_, const stride_type* cscat_a_,
                     void* p_ap_)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KE = Config::template gemm_kr<T>::extent;

    T alpha = *static_cast<const T*>(alpha_);

    const T* TBLIS_RESTRICT p_a  = static_cast<const T*>(p_a_ );
          T* TBLIS_RESTRICT p_ap = static_cast<      T*>(p_ap_);

    const stride_type* TBLIS_RESTRICT rscat_a  = rscat_a_;
    const stride_type* TBLIS_RESTRICT cscat_a  = cscat_a_;

    if (conj && is_complex<T>::value)
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
                p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rscat_a[mr] + cscat_a[p]]);

            for (len_type mr = m;mr < MR;mr++)
                p_ap[mr + ME*p] = T();
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
                p_ap[mr + ME*p] = alpha * p_a[rscat_a[mr] + cscat_a[p]];

            for (len_type mr = m;mr < MR;mr++)
                p_ap[mr + ME*p] = T();
        }
    }

    for (len_type p = k;p < round_up(k,KE);p++)
        for (len_type mr = 0;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
}

template <typename Config, typename T, int Mat>
void pack_nb_ukr_def(len_type m, len_type k, const void* alpha_, bool conj,
                     const void* p_a_, stride_type rs_a,
                     const stride_type* cscat_a_, const stride_type* cbs_a_,
                     void* p_ap_)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;
    constexpr len_type KE = Config::template gemm_kr<T>::extent;

    T alpha = *static_cast<const T*>(alpha_);

    const T* TBLIS_RESTRICT p_a  = static_cast<const T*>(p_a_ );
          T* TBLIS_RESTRICT p_ap = static_cast<      T*>(p_ap_);

    const stride_type* TBLIS_RESTRICT cscat_a  = cscat_a_;
    const stride_type* TBLIS_RESTRICT cbs_a  = cbs_a_;

    if (conj && is_complex<T>::value)
    {
        if (m == MR && rs_a == 1)
        {
            len_type p = 0;
            for (;p < k-KR;p += KR)
            {
                const stride_type cs_a = cbs_a[p];

                if (cs_a)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        #pragma omp simd
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[mr + cscat_a[p] + cs_a*kr]);
                }
                else
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[mr + cscat_a[p+kr]]);
                }
            }

            for (;p < k;p++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*p] = alpha * tblis::conj(p_a[mr + cscat_a[p]]);
        }
        else if (m == MR)
        {
            len_type p = 0;
            for (;p < k-KR;p += KR)
            {
                const stride_type cs_a = cbs_a[p];

                if (cs_a)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[rs_a*mr + cscat_a[p] + cs_a*kr]);
                }
                else
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * tblis::conj(p_a[rs_a*mr + cscat_a[p+kr]]);
                }
            }

            for (;p < k;p++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + cscat_a[p]]);
        }
        else
        {
            for (len_type p = 0;p < k;p++)
            {
                for (len_type mr = 0;mr < m;mr++)
                    p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rs_a*mr + cscat_a[p]]);

                for (len_type mr = m;mr < MR;mr++)
                    p_ap[mr + ME*p] = T();
            }
        }
    }
    else
    {
        if (m == MR && rs_a == 1)
        {
            len_type p = 0;
            for (;p < k-KR;p += KR)
            {
                const stride_type cs_a = cbs_a[p];

                if (cs_a)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        #pragma omp simd
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * p_a[mr + cscat_a[p] + cs_a*kr];
                }
                else
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * p_a[mr + cscat_a[p+kr]];
                }
            }

            for (;p < k;p++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*p] = alpha * p_a[mr + cscat_a[p]];
        }
        else if (m == MR)
        {
            len_type p = 0;
            for (;p < k-KR;p += KR)
            {
                const stride_type cs_a = cbs_a[p];

                if (cs_a)
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * p_a[rs_a*mr + cscat_a[p] + cs_a*kr];
                }
                else
                {
                    for (len_type kr = 0;kr < KR;kr++)
                        for (len_type mr = 0;mr < MR;mr++)
                            p_ap[mr + ME*(p+kr)] = alpha * p_a[rs_a*mr + cscat_a[p+kr]];
                }
            }

            for (;p < k;p++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + cscat_a[p]];
        }
        else
        {
            for (len_type p = 0;p < k;p++)
            {
                for (len_type mr = 0;mr < m;mr++)
                    p_ap[mr + ME*p] = alpha * p_a[rs_a*mr + cscat_a[p]];

                for (len_type mr = m;mr < MR;mr++)
                    p_ap[mr + ME*p] = T();
            }
        }
    }

    for (len_type p = k;p < round_up(k,KE);p++)
        for (len_type mr = 0;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
}

template <typename Config, typename T, int Mat>
void pack_ss_scal_ukr_def(len_type m, len_type k, const void* alpha_, bool conj,
                          const void* p_a_,
                          const stride_type* rscat_a_, const void* rscale_a_,
                          const stride_type* cscat_a_, const void* cscale_a_,
                          void* p_ap_)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KE = Config::template gemm_kr<T>::extent;

    T alpha = *static_cast<const T*>(alpha_);

    const T* TBLIS_RESTRICT p_a      = static_cast<const T*>(p_a_     );
          T* TBLIS_RESTRICT p_ap     = static_cast<      T*>(p_ap_    );
    const T* TBLIS_RESTRICT rscale_a = static_cast<const T*>(rscale_a_);
    const T* TBLIS_RESTRICT cscale_a = static_cast<const T*>(cscale_a_);

    const stride_type* TBLIS_RESTRICT rscat_a  = rscat_a_;
    const stride_type* TBLIS_RESTRICT cscat_a  = cscat_a_;

    if (conj && is_complex<T>::value)
    {
        if (m == MR)
        {
            for (len_type p = 0;p < k;p++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rscat_a[mr] + cscat_a[p]]) * rscale_a[mr] * cscale_a[p];
        }
        else
        {
            for (len_type p = 0;p < k;p++)
            {
                for (len_type mr = 0;mr < m;mr++)
                    p_ap[mr + ME*p] = alpha * tblis::conj(p_a[rscat_a[mr] + cscat_a[p]]) * rscale_a[mr] * cscale_a[p];

                for (len_type mr = m;mr < MR;mr++)
                    p_ap[mr + ME*p] = T();
            }
        }
    }
    else
    {
        if (m == MR)
        {
            for (len_type p = 0;p < k;p++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*p] = alpha * p_a[rscat_a[mr] + cscat_a[p]] * rscale_a[mr] * cscale_a[p];
        }
        else
        {
            for (len_type p = 0;p < k;p++)
            {
                for (len_type mr = 0;mr < m;mr++)
                    p_ap[mr + ME*p] = alpha * p_a[rscat_a[mr] + cscat_a[p]] * rscale_a[mr] * cscale_a[p];

                for (len_type mr = m;mr < MR;mr++)
                    p_ap[mr + ME*p] = T();
            }
        }
    }

    for (len_type p = k;p < round_up(k,KE);p++)
        for (len_type mr = 0;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
}

}

#endif //TBLIS_KERNELS_3M_PACKM_HPP
