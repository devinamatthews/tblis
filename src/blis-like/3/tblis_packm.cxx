#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a, inc_t rs_a, inc_t cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        if (rs_a == 1)
        {
            for (dim_t i = 0;i < k;i++)
            {
                for (dim_t mr = 0;mr < MR;mr++)
                {
                    p_ap[mr] = p_a[mr];
                }

                p_a += cs_a;
                p_ap += MR;
            }
        }
        else if (cs_a == 1)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                for (dim_t i = 0;i < k;i++)
                {
                    p_ap[MR*i+mr] = p_a[i+rs_a*mr];
                }
            }

            p_a += cs_a*k;
            p_ap += MR*k;
        }
        else
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
                    const T* restrict & p_a, const inc_t* restrict & rs_a, inc_t cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[rs_a[mr]];
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
                p_ap[mr] = p_a[rs_a[mr]];
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
                    const T* restrict & p_a, inc_t rs_a, const inc_t* restrict cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a*mr];
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
                p_ap[mr] = p_a[cs_a[i] + rs_a*mr];
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
                    const T* restrict & p_a, const inc_t* restrict & rs_a, const inc_t* restrict cs_a,
                    T* restrict & p_ap)
{
    dim_t k_rem = detail::remainder(k, KR);

    if (m == MR)
    {
        for (dim_t i = 0;i < k;i++)
        {
            for (dim_t mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a[mr]];
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
                p_ap[mr] = p_a[cs_a[i] + rs_a[mr]];
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
void PackRowPanel<T,MR,KR,Trans>::operator()(const Matrix<T>& A, Matrix<T>& Ap) const
{
    dim_t m_a = (Trans ? A.width () : A.length());
    dim_t k_a = (Trans ? A.length() : A.width ());
    inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
    inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
    const T* p_a = A.data();
    T* p_ap = Ap.data();

    for (dim_t off_m = 0;off_m < m_a;off_m += MR)
    {
        PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                p_a, rs_a, cs_a, p_ap);
    }
}

template <typename T, dim_t MR, dim_t KR, bool Trans>
void PackRowPanel<T,MR,KR,Trans>::operator()(const ScatterMatrix<T>& A, Matrix<T>& Ap) const
{
    dim_t m_a = (Trans ? A.width () : A.length());
    dim_t k_a = (Trans ? A.length() : A.width ());
    inc_t rs_a = (Trans ? A.col_stride() : A.row_stride());
    inc_t cs_a = (Trans ? A.row_stride() : A.col_stride());
    const inc_t* rscat_a = (Trans ? A.col_scatter() : A.row_scatter());
    const inc_t* cscat_a = (Trans ? A.row_scatter() : A.col_scatter());
    const T* p_a = A.data();
    T* p_ap = Ap.data();

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

template <typename T, dim_t MR, dim_t KR, bool Trans>
template <bool _Trans>
typename std::enable_if<_Trans == false>::type
PackRowPanel<T,MR,KR,Trans>::operator()(const BlockScatterMatrix<T,MR,0>& A_, Matrix<T>& Ap) const
{
    BlockScatterMatrix<T,MR,0> A(A_);

    dim_t m_a = A.length();
    dim_t k_a = A.width();
    T* p_ap = Ap.data();

    A.length(MR);

    for (dim_t off_m = 0;off_m < m_a;off_m += MR)
    {
        inc_t rs_a = A.row_stride();
        inc_t cs_a = A.col_stride();
        const inc_t* rscat_a = A.row_scatter();
        const inc_t* cscat_a = A.col_scatter();
        const T* p_a = A.data();

        if (rs_a == 0 && cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cscat_a, p_ap);
        }
        else if (rs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cs_a, p_ap);
        }
        else if (cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cscat_a, p_ap);
        }
        else
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cs_a, p_ap);
        }

        A.shift_down();
    }

    A.length(m_a);
    A.shift_up();
}

template <typename T, dim_t MR, dim_t KR, bool Trans>
template <bool _Trans>
typename std::enable_if<_Trans == true>::type
PackRowPanel<T,MR,KR,Trans>::operator()(const BlockScatterMatrix<T,0,MR>& A_, Matrix<T>& Ap) const
{
    BlockScatterMatrix<T,0,MR> A(A_);

    dim_t m_a = A.width();
    dim_t k_a = A.length();
    T* p_ap = Ap.data();

    A.width(MR);

    for (dim_t off_m = 0;off_m < m_a;off_m += MR)
    {
        inc_t rs_a = A.col_stride();
        inc_t cs_a = A.row_stride();
        const inc_t* rscat_a = A.col_scatter();
        const inc_t* cscat_a = A.row_scatter();
        const T* p_a = A.data();

        if (rs_a == 0 && cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cscat_a, p_ap);
        }
        else if (rs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rscat_a, cs_a, p_ap);
        }
        else if (cs_a == 0)
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cscat_a, p_ap);
        }
        else
        {
            PackMicroPanel<T,MR,KR>(std::min(MR, m_a-off_m), k_a,
                                    p_a, rs_a, cs_a, p_ap);
        }

        A.shift_right();
    }

    A.width(m_a);
    A.shift_left();
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T*& p_a, inc_t rs_a, inc_t cs_a, T*& p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T*& p_a, inc_t rs_a, inc_t cs_a, T*& p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T*& p_a, const inc_t*& rs_a, inc_t cs_a, T*& p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T*& p_a, const inc_t*& rs_a, inc_t cs_a, T*& p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T*& p_a, inc_t rs_a, const inc_t* cs_a, T*& p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T*& p_a, inc_t rs_a, const inc_t* cs_a, T*& p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void PackMicroPanel<T,          MR, KR>(dim_t m, dim_t k, const T*& p_a, const inc_t*& rs_a, const inc_t* cs_a, T*& p_ap); \
template void PackMicroPanel<T,(MR==NR?0:NR),KR>(dim_t m, dim_t k, const T*& p_a, const inc_t*& rs_a, const inc_t* cs_a, T*& p_ap);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template struct PackRowPanel<T,MR,KR,false>; \
template struct PackRowPanel<T,NR,KR, true>; \
template std::enable_if<true>::type PackRowPanel<T,MR,KR,false>::operator()<false>(const BlockScatterMatrix<T,MR,0>& A, Matrix<T>& Ap) const; \
template std::enable_if<true>::type PackRowPanel<T,NR,KR, true>::operator()< true>(const BlockScatterMatrix<T,0,NR>& A, Matrix<T>& Ap) const;
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
