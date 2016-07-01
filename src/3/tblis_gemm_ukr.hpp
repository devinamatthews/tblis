#ifndef _TBLIS_GEMM_UKR_HPP_
#define _TBLIS_GEMM_UKR_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T, idx_type MR, idx_type NR>
void AccumulateMicroTile(idx_type m, idx_type n, const T* restrict p_ab,
                         T beta, T* restrict p_c, stride_type rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR] + beta*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR>
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
                p_c[rs_c[i] + j*cs_c] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i + j*MR] + beta*p_c[rs_c[i] + j*cs_c];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR>
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
                p_c[i*rs_c + cs_c[j]] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i + j*MR] + beta*p_c[i*rs_c + cs_c[j]];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR>
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
                p_c[rs_c[i] + cs_c[j]] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (idx_type j = 0;j < n;j++)
        {
            for (idx_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i + j*MR] + beta*p_c[rs_c[i] + cs_c[j]];
            }
        }
    }
}

template <typename T, idx_type MR, idx_type NR>
void GenericMicroKernel(idx_type k,
                        const T* restrict alpha,
                        const T* restrict p_a, const T* restrict p_b,
                        const T* restrict beta,
                        T* restrict p_c, stride_type rs_c, stride_type cs_c,
                        const void* restrict auxinfo, const void* restrict cntx)
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

    AccumulateMicroTile<T,MR,NR>(MR, NR, p_ab, *beta, p_c, rs_c, cs_c);
}

template <typename Config>
struct MicroKernel
{
    template <typename T>
    struct run
    {
        constexpr static idx_type MR = Config::template MR<T>::def;
        constexpr static idx_type NR = Config::template NR<T>::def;
        constexpr static gemm_ukr_t<T> ukr = Config::template gemm_ukr<T>::value;

        void operator()(ThreadCommunicator& comm,
                        T alpha, const const_matrix_view<T>& A,
                                 const const_matrix_view<T>& B,
                        T  beta,             matrix_view<T>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            idx_type m = C.length();
            idx_type n = C.width();
            idx_type k = A.width();
            stride_type rs_c = C.row_stride();
            stride_type cs_c = C.col_stride();

            if (m == MR && n == NR)
            {
                ukr(k, &alpha, p_a, p_b,
                    &beta, p_c, rs_c, cs_c,
                    nullptr, nullptr);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(64)));
                static constexpr T zero = 0.0;

                ukr(k, &alpha, p_a, p_b,
                    &zero, p_ab, 1, MR,
                    nullptr, nullptr);

                AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                             beta, p_c, rs_c, cs_c);
            }
        }

        void operator()(ThreadCommunicator& comm,
                        T alpha, const const_matrix_view<T>& A,
                                 const const_matrix_view<T>& B,
                        T  beta,     scatter_matrix_view<T>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            idx_type m = C.length();
            idx_type n = C.width();
            idx_type k = A.width();
            stride_type rs_c = C.row_stride();
            stride_type cs_c = C.col_stride();
            const stride_type* rscat_c = C.row_scatter();
            const stride_type* cscat_c = C.col_scatter();

            if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
            {
                ukr(k, &alpha, p_a, p_b,
                    &beta, p_c, rs_c, cs_c,
                    nullptr, nullptr);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(64)));
                static constexpr T zero = 0.0;

                ukr(k, &alpha, p_a, p_b,
                    &zero, p_ab, 1, MR,
                    nullptr, nullptr);

                if (rs_c == 0 && cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cscat_c);
                }
                else if (rs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cs_c);
                }
                else if (cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rs_c, cscat_c);
                }
                else
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rs_c, cs_c);
                }
            }
        }

        void operator()(ThreadCommunicator& comm,
                        T alpha,    const const_matrix_view<T>& A,
                                    const const_matrix_view<T>& B,
                        T  beta, block_scatter_matrix<T,MR,NR>& C) const
        {
            const T* p_a = A.data();
            const T* p_b = B.data();
                  T* p_c = C.data();

            idx_type m = C.length();
            idx_type n = C.width();
            idx_type k = A.width();
            stride_type rs_c = C.row_stride();
            stride_type cs_c = C.col_stride();
            const stride_type* rscat_c = C.row_scatter();
            const stride_type* cscat_c = C.col_scatter();

            if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
            {
                ukr(k, &alpha, p_a, p_b,
                    &beta, p_c, rs_c, cs_c,
                    nullptr, nullptr);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(64)));
                static constexpr T zero = 0.0;

                ukr(k, &alpha, p_a, p_b,
                    &zero, p_ab, 1, MR,
                    nullptr, nullptr);

                if (rs_c == 0 && cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cscat_c);
                }
                else if (rs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rscat_c, cs_c);
                }
                else if (cs_c == 0)
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rs_c, cscat_c);
                }
                else
                {
                    AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                                 beta, p_c, rs_c, cs_c);
                }
            }
        }
    };
};

}

#endif
