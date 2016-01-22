#ifndef _TENSOR_TBLIS_UKERNEL_HPP_
#define _TENSOR_TBLIS_UKERNEL_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, T* restrict p_ab,
                         T beta, T* restrict p_c, inc_t rs_c, inc_t cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + i*rs_c + j*cs_c) = *(p_ab + i + j*MR);
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + i*rs_c + j*cs_c) = *(p_ab + i + j*MR) + beta*(*(p_c + i*rs_c + j*cs_c));
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, T* restrict p_ab,
                         T beta, T* restrict p_c, inc_t* restrict rs_c, inc_t cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + rs_c[i] + j*cs_c) = *(p_ab + i + j*MR);
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + rs_c[i] + j*cs_c) = *(p_ab + i + j*MR) + beta*(*(p_c + rs_c[i] + j*cs_c));
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, T* restrict p_ab,
                         T beta, T* restrict p_c, inc_t rs_c, inc_t* restrict cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + i*rs_c + cs_c[j]) = *(p_ab + i + j*MR);
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + i*rs_c + cs_c[j]) = *(p_ab + i + j*MR) + beta*(*(p_c + i*rs_c + cs_c[j]));
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, T* restrict p_ab,
                         T beta, T* restrict p_c, inc_t* restrict rs_c, inc_t* restrict cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + rs_c[i] + cs_c[j]) = *(p_ab + i + j*MR);
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                *(p_c + rs_c[i] + cs_c[j]) = *(p_ab + i + j*MR) + beta*(*(p_c + rs_c[i] + cs_c[j]));
            }
        }
    }
}

template <template <typename> class MT, template <typename> class NT>
struct MicroKernel
{
    template <typename T>
    struct run
    {
        void operator()(T alpha, Matrix<T>& A, Matrix<T>& B, T beta, Matrix<T>& C) const
        {
            constexpr dim_t MR = MT<T>::value;
            constexpr dim_t NR = NT<T>::value;

            T* p_a = A;
            T* p_b = B;
            T* p_c = C;

            dim_t m = C.length();
            dim_t n = C.width();
            dim_t k = A.width();
            inc_t rs_c = C.row_stride();
            inc_t cs_c = C.col_stride();

            auxinfo_t data;
            bli_auxinfo_set_next_ab(p_a, p_b, data);

            if (m == MR && n == NR)
            {
                gemm_ukr_t<T>::value(k,
                                     &alpha, p_a, p_b,
                                     &beta, p_c, rs_c, cs_c,
                                     &data);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
                static constexpr T zero = 0.0;

                gemm_ukr_t<T>::value(k,
                                     &alpha, p_a, p_b,
                                     (T*)&zero, p_ab, 1, MR,
                                     &data);

                AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                             beta, p_c, rs_c, cs_c);
            }
        }

        void operator()(T alpha, Matrix<T>& A, Matrix<T>& B, T beta, ScatterMatrix<T>& C) const
        {
            constexpr dim_t MR = MT<T>::value;
            constexpr dim_t NR = NT<T>::value;

            T* p_a = A;
            T* p_b = B;
            T* p_c = C;

            dim_t m = C.length();
            dim_t n = C.width();
            dim_t k = A.width();
            inc_t rs_c = C.row_stride();
            inc_t cs_c = C.col_stride();
            inc_t* rscat_c = C.row_scatter();
            inc_t* cscat_c = C.col_scatter();

            auxinfo_t data;
            bli_auxinfo_set_next_ab(p_a, p_b, data);

            if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
            {
                gemm_ukr_t<T>::value(k,
                                     &alpha, p_a, p_b,
                                     &beta, p_c, rs_c, cs_c,
                                     &data);
            }
            else
            {
                T p_ab[MR*NR] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
                static constexpr T zero = 0.0;

                gemm_ukr_t<T>::value(k,
                                     &alpha, p_a, p_b,
                                     (T*)&zero, p_ab, 1, MR,
                                     &data);

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
