#ifndef _TENSOR_TBLIS_PACK_HPP_
#define _TENSOR_TBLIS_PACK_HPP_

#include "tblis.hpp"
#include "normfm.hpp"

namespace tblis
{

namespace detail
{
    constexpr inline dim_t remainder(dim_t N, dim_t B)
    {
        return (B-1)-(N+B-1)%B;
    }

    constexpr inline dim_t round_up(dim_t N, dim_t B)
    {
        return (N+B-1)-(N+B-1)%B;
    }
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    T* restrict & p_a, inc_t rs_a, inc_t cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[rs_a*mr];
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[rs_a*mr];
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a += rs_a*m - cs_a*k;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    T* restrict & p_a, inc_t* restrict & rs_a, inc_t cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = *(p_a+rs_a[mr]);
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = *(p_a+rs_a[mr]);
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a -= cs_a*k;
    rs_a += m;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    T* restrict & p_a, inc_t rs_a, inc_t* restrict cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = (p_a+cs_a[i])[rs_a*mr];
            }

            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = (p_a+cs_a[i])[rs_a*mr];
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a += rs_a*m;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    T* restrict & p_a, inc_t* restrict & rs_a, inc_t* restrict cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = *(p_a+cs_a[i]+rs_a[mr]);
            }

            p_ap += MR;
        }
    }
    else
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < m;mr++)
            {
                p_ap[mr] = *(p_a+cs_a[i]+rs_a[mr]);
            }

            for (dim_t mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_ap += MR;
        }
    }

    for (dim_t i = 0;i < k_rem;i++)
    {
        for (dim_t mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    rs_a += m;
}

template <typename T, dim_t MR, dim_t KR, bool Trans>
struct PackRowPanel
{
    void operator()(Matrix<T>& A, Matrix<T>& Ap) const
    {
        dim_t m_a = (Trans ? A.width () : A.length());
        dim_t k_a = (Trans ? A.length() : A.width ());
        inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
        inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
        T* p_a = A;
        T* p_ap = Ap;

        for (dim_t off_m = 0;off_m < m_a;off_m += MR)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cs_a, p_ap);
        }
    }

    void operator()(ScatterMatrix<T>& A, Matrix<T>& Ap) const
    {
        dim_t m_a = (Trans ? A.width () : A.length());
        dim_t k_a = (Trans ? A.length() : A.width ());
        inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
        inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
        inc_t* rscat_a = (Trans ? A.col_scatter() : A.row_scatter());
        inc_t* cscat_a = (Trans ? A.row_scatter() : A.col_scatter());
        T* p_a = A;
        T* p_ap = Ap;

        if (rs_a == 0 && cs_a == 0)
        {
            for (dim_t off_m = 0;off_m < m_a;off_m += MR)
            {
                PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                        p_a, rscat_a, cscat_a, p_ap);
            }
        }
        else if (rs_a == 0)
        {
            for (dim_t off_m = 0;off_m < m_a;off_m += MR)
            {
                PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                        p_a, rscat_a, cs_a, p_ap);
            }
        }
        else if (cs_a == 0)
        {
            for (dim_t off_m = 0;off_m < m_a;off_m += MR)
            {
                PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                        p_a, rs_a, cscat_a, p_ap);
            }
        }
        else
        {
            for (dim_t off_m = 0;off_m < m_a;off_m += MR)
            {
                PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                        p_a, rs_a, cs_a, p_ap);
            }
        }
    }
};

template <typename Pack, typename Run, int Mat>
struct PackAndRun
{
    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(A, P);
        Run()(alpha, P, B, beta, C);
    }
};

template <typename Pack, typename Run>
struct PackAndRun<Pack, Run, matrix_constants::MAT_B>
{
    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(B, P);
        Run()(alpha, A, P, beta, C);
    }
};

template <template <typename> class MT, template <typename> class KT, int Mat>
struct Pack
{
    template <typename T, typename Child, typename... Children>
    struct run
    {
        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C) const
        {
            using namespace matrix_constants;

            constexpr dim_t MR = MT<T>::value;
            constexpr dim_t KR = KT<T>::value;

            constexpr packbuf_t PackBuf = (Mat == MAT_A ? BLIS_BUFFER_FOR_A_BLOCK
                                                        : BLIS_BUFFER_FOR_B_PANEL);
            constexpr bool Trans = (Mat == MAT_B);

            typedef PackRowPanel<T,MR,KR,Trans> Pack;
            typedef typename Child::template run<T, Children...> Run;

            dim_t m_p = (Mat == MAT_A ? A.length() : B.width ());
            dim_t k_p = (Mat == MAT_A ? A.width () : B.length());
            m_p = detail::round_up(m_p, MR);
            k_p = detail::round_up(k_p, KR);
            PooledMemory<T> buf(m_p*k_p, PackBuf);

            Matrix<T> P((Mat == MAT_A ? m_p : k_p),
                        (Mat == MAT_A ? k_p : m_p),
                        buf,
                        (Mat == MAT_A ? k_p :   1),
                        (Mat == MAT_A ?   1 : k_p));

            PackAndRun<Pack,Run,Mat>(alpha, A, B, beta, C, P);
        }
    };
};

template <template <typename> class MT, template <typename> class KT>
using PackA = Pack<MT,KT,matrix_constants::MAT_A>;

template <template <typename> class NT, template <typename> class KT>
using PackB = Pack<NT,KT,matrix_constants::MAT_B>;

}

#endif
