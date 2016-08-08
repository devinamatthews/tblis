#ifndef _TBLIS_GEMM_UKR_HPP_
#define _TBLIS_GEMM_UKR_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T, idx_type MR, idx_type NR, idx_type RS, idx_type CS>
void AccumulateMicroTile(idx_type m, idx_type n, const T* restrict p_ab,
                         T beta, T* restrict p_c, stride_type rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*RS + j*CS] + beta*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR, idx_type RS, idx_type CS>
void AccumulateMicroTile(idx_type m, idx_type n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         const stride_type* restrict rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*RS + j*CS] + beta*p_c[rs_c[i] + j*cs_c];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR, idx_type RS, idx_type CS>
void AccumulateMicroTile(idx_type m, idx_type n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         stride_type rs_c, const stride_type* restrict cs_c)
{
    if (beta == T(0))
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*RS + j*CS] + beta*p_c[i*rs_c + cs_c[j]];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR, idx_type RS, idx_type CS>
void AccumulateMicroTile(idx_type m, idx_type n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         const stride_type* restrict rs_c,
                         const stride_type* restrict cs_c)
{
    if (beta == T(0))
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*RS + j*CS];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*RS + j*CS] + beta*p_c[rs_c[i] + cs_c[j]];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR>
void GenericMicroKernel(stride_type k,
                        const T* restrict alpha,
                        const T* restrict p_a, const T* restrict p_b,
                        const T* restrict beta,
                        T* restrict p_c, stride_type rs_c, stride_type cs_c)
{
    T p_ab[MR*NR] __attribute__((aligned(64))) = {};

    while (k --> 0)
    {
        for (int j = 0;j < NR;j++)
        {
            for (int i = 0;i < MR;i++)
            {
                p_ab[i + MR*j] += p_a[i] * p_b[j];
            }
        }

        p_a += MR;
        p_b += NR;
    }

    for (int j = 0;j < NR;j++)
    {
        for (int i = 0;i < MR;i++)
        {
            p_ab[i + MR*j] *= *alpha;
        }
    }

    AccumulateMicroTile<T,MR,NR,1,MR>(MR, NR, p_ab, *beta, p_c, rs_c, cs_c);
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
        constexpr static idx_type MR = Config::template MR<T>::def;
        constexpr static idx_type NR = Config::template NR<T>::def;
        constexpr static gemm_ukr_t<T> ukr = Config::template gemm_ukr<T>::value;
        constexpr static bool row_major = Config::template gemm_row_major<T>::value;
        constexpr static idx_type RS = (row_major ? NR : 1);
        constexpr static idx_type CS = (row_major ? 1 : MR);

        void operator()(const gemm_thread_config& cfg, thread_communicator& comm,
                        T alpha, matrix_view<T>& A,
                                 matrix_view<T>& B,
                        T  beta, matrix_view<T>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            idx_type m = C.length(0);
            idx_type n = C.length(1);
            idx_type k = A.length(1);
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

            idx_type m = C.length(0);
            idx_type n = C.length(1);
            idx_type k = A.length(1);
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

            idx_type m = C.length(0);
            idx_type n = C.length(1);
            idx_type k = A.length(1);
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
