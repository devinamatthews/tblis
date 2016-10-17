#ifndef _TBLIS_GEMM_UKR_HPP_
#define _TBLIS_GEMM_UKR_HPP_

#include "../../tblis_config.h"
#include "../configs/configs.hpp.in"
#include "../util/basic_types.h"

namespace tblis
{

template <typename T, len_type MR, len_type NR, len_type RS, len_type CS>
void AccumulateMicroTile(len_type m, len_type n, const T* TBLIS_RESTRICT p_ab,
                         T beta, T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*RS + j*CS] + beta*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

template <typename T, len_type MR, len_type NR, len_type RS, len_type CS>
void AccumulateMicroTile(len_type m, len_type n, const T* TBLIS_RESTRICT p_ab,
                         T beta, T* TBLIS_RESTRICT p_c,
                         const stride_type* TBLIS_RESTRICT rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*RS + j*CS] + beta*p_c[rs_c[i] + j*cs_c];
            }
        }
    }
}

template <typename T, len_type MR, len_type NR, len_type RS, len_type CS>
void AccumulateMicroTile(len_type m, len_type n, const T* TBLIS_RESTRICT p_ab,
                         T beta, T* TBLIS_RESTRICT p_c,
                         stride_type rs_c, const stride_type* TBLIS_RESTRICT cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*RS + j*CS] + beta*p_c[i*rs_c + cs_c[j]];
            }
        }
    }
}

template <typename T, len_type MR, len_type NR, len_type RS, len_type CS>
void AccumulateMicroTile(len_type m, len_type n, const T* TBLIS_RESTRICT p_ab,
                         T beta, T* TBLIS_RESTRICT p_c,
                         const stride_type* TBLIS_RESTRICT rs_c,
                         const stride_type* TBLIS_RESTRICT cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*RS + j*CS] + beta*p_c[rs_c[i] + cs_c[j]];
            }
        }
    }
}

template <typename Config>
struct MicroKernel
{
    struct auxinfo_t
    {
        int pack_a, pack_b;
        const void*  a_next;
        const void*  b_next;
    };

    template <typename T>
    struct run
    {
        constexpr static len_type MR = config_traits<Config>::template MR<T>::def;
        constexpr static len_type NR = config_traits<Config>::template NR<T>::def;
        constexpr static gemm_ukr_t<T> ukr = config_traits<Config>::template gemm_ukr<T>::value;
        constexpr static bool row_major = config_traits<Config>::template gemm_row_major<T>::value;
        constexpr static len_type RS = (row_major ? NR : 1);
        constexpr static len_type CS = (row_major ? 1 : MR);

        void operator()(const gemm_thread_config& cfg, thread_communicator& comm,
                        T alpha, matrix_view<T>& A,
                                 matrix_view<T>& B,
                        T  beta, matrix_view<T>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            len_type m = C.length(0);
            len_type n = C.length(1);
            len_type k = A.length(1);
            stride_type rs_c = C.stride(0);
            stride_type cs_c = C.stride(1);

            auxinfo_t aux{0, 0, p_a, p_b};

            if (m == MR && n == NR)
            {
                ukr((long)k, &alpha, p_a, p_b,
                    &beta, p_c, rs_c, cs_c,
                    &aux, nullptr);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(64)));
                static constexpr T zero = 0.0;

                ukr((long)k, &alpha, p_a, p_b,
                    &zero, p_ab, RS, CS,
                    &aux, nullptr);

                AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                             beta, p_c, rs_c, cs_c);
            }

            //printf_locked("%p %f -- %d %d %d %ld %ld %f %f %p %p %f %f\n", p_c, pow((double)real(tblis_normfm(m, n, p_c, rs_c, cs_c)),2),
            //              m, n, k, rs_c, cs_c, (double)real(alpha), (double)real(beta),
            //              p_a, p_b, pow((double)real(tblis_normfv(MR*k, p_a, 1)),2),
            //              pow((double)real(tblis_normfv(NR*k, p_b, 1)),2));
        }

        void operator()(const gemm_thread_config& cfg, thread_communicator& comm,
                        T alpha,         matrix_view<T>& A,
                                         matrix_view<T>& B,
                        T  beta, scatter_matrix_view<T>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            len_type m = C.length(0);
            len_type n = C.length(1);
            len_type k = A.length(1);
            stride_type rs_c = C.stride(0);
            stride_type cs_c = C.stride(1);
            const stride_type* rscat_c = C.scatter(0);
            const stride_type* cscat_c = C.scatter(1);

            auxinfo_t aux{0, 0, p_a, p_b};

            if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
            {
                ukr(k, &alpha, p_a, p_b,
                    &beta, p_c, rs_c, cs_c,
                    &aux, nullptr);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(64)));
                static constexpr T zero = 0.0;

                ukr(k, &alpha, p_a, p_b,
                    &zero, p_ab, RS, CS,
                    &aux, nullptr);

                if (rs_c == 0 && cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cscat_c);
                }
                else if (rs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cs_c);
                }
                else if (cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rs_c, cscat_c);
                }
                else
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rs_c, cs_c);
                }
            }
        }

        void operator()(const gemm_thread_config& cfg, thread_communicator& comm,
                        T alpha,                matrix_view<T>& A,
                                                matrix_view<T>& B,
                        T  beta, block_scatter_matrix<T,MR,NR>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            len_type m = C.length(0);
            len_type n = C.length(1);
            len_type k = A.length(1);
            stride_type rs_c = C.stride(0);
            stride_type cs_c = C.stride(1);
            const stride_type* rscat_c = C.scatter(0);
            const stride_type* cscat_c = C.scatter(1);

            auxinfo_t aux{0, 0, p_a, p_b};

            if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
            {
                ukr(k, &alpha, p_a, p_b,
                    &beta, p_c, rs_c, cs_c,
                    &aux, nullptr);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(64)));
                static constexpr T zero = 0.0;

                ukr(k, &alpha, p_a, p_b,
                    &zero, p_ab, RS, CS,
                    &aux, nullptr);

                if (rs_c == 0 && cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cscat_c);
                }
                else if (rs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cs_c);
                }
                else if (cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rs_c, cscat_c);
                }
                else
                {
                    AccumulateMicroTile<T,MR,NR,RS,CS>(m, n, p_ab,
                                                 beta, p_c, rs_c, cs_c);
                }
            }
        }
    };
};

}

#endif
