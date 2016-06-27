#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c, inc_t rs_c, inc_t cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i + j*MR] + beta*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         const inc_t* restrict rs_c, inc_t cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i + j*MR] + beta*p_c[rs_c[i] + j*cs_c];
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         inc_t rs_c, const inc_t* restrict cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i + j*MR] + beta*p_c[i*rs_c + cs_c[j]];
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void AccumulateMicroTile(dim_t m, dim_t n, const T* restrict p_ab,
                         T beta, T* restrict p_c,
                         const inc_t* restrict rs_c,
                         const inc_t* restrict cs_c)
{
    if (beta == 0.0)
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i + j*MR];
            }
        }
    }
    else
    {
        for (dim_t j = 0;j < n;j++)
        {
            for (dim_t i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i + j*MR] + beta*p_c[rs_c[i] + cs_c[j]];
            }
        }
    }
}

template <typename T, dim_t MR, dim_t NR>
void GenericMicroKernel(dim_t k,
                        const T* restrict alpha,
                        const T* restrict p_a, const T* restrict p_b,
                        const T* restrict beta,
                        T* restrict p_c, inc_t rs_c, inc_t cs_c,
                        const void* restrict data, const void* restrict cntx)
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
            p_ab[i + MR*j] *= alpha;
        }
    }

    AccumulateMicroTile<T,MR,NR>(MR, NR, p_ab, beta, p_c, rs_c, cs_c);
}

template <typename Config>
template <typename T>
void MicroKernel<Config>::run<T>::operator()(ThreadCommunicator& comm,
                                             T alpha, const const_matrix_view<T>& A,
                                                      const const_matrix_view<T>& B,
                                             T  beta,             matrix_view<T>& C) const
{
    const T* p_a = A.data();
    const T* p_b = B.data();
          T* p_c = C.data();

    dim_t m = C.length();
    dim_t n = C.width();
    dim_t k = A.width();
    inc_t rs_c = C.row_stride();
    inc_t cs_c = C.col_stride();

    auxinfo_t data;
    //bli_auxinfo_set_next_ab(p_a, p_b, data);

    if (m == MR && n == NR)
    {
        ukr(k, &alpha, p_a, p_b,
            &beta, p_c, rs_c, cs_c,
            &data, nullptr);
    }
    else
    {
        T p_ab[MR*NR] __attribute__((aligned(64)));
        static constexpr T zero = 0.0;

        ukr(k, &alpha, p_a, p_b,
            &zero, p_ab, 1, MR,
            &data, nullptr);

        AccumulateMicroTile<T,MR,NR>(m, n, p_ab,
                                     beta, p_c, rs_c, cs_c);
    }
}

template <typename Config>
template <typename T>
void MicroKernel<Config>::run<T>::operator()(ThreadCommunicator& comm,
                                             T alpha, const const_matrix_view<T>& A,
                                                      const const_matrix_view<T>& B,
                                             T  beta,           const_scatter_matrix_view<T>& C) const
{
    const T* p_a = A.data();
    const T* p_b = B.data();
          T* p_c = C.data();

    dim_t m = C.length();
    dim_t n = C.width();
    dim_t k = A.width();
    inc_t rs_c = C.row_stride();
    inc_t cs_c = C.col_stride();
    const inc_t* rscat_c = C.row_scatter();
    const inc_t* cscat_c = C.col_scatter();

    auxinfo_t data;
    //bli_auxinfo_set_next_ab(p_a, p_b, data);

    if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
    {
        ukr(k, &alpha, p_a, p_b,
            &beta, p_c, rs_c, cs_c,
            &data, nullptr);
    }
    else
    {
        T p_ab[MR*NR] __attribute__((aligned(64)));
        static constexpr T zero = 0.0;

        ukr(k, &alpha, p_a, p_b,
            &zero, p_ab, 1, MR,
            &data, nullptr);

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

template <typename Config>
template <typename T>
void MicroKernel<Config>::run<T>::operator()(ThreadCommunicator& comm,
                                             T alpha,  const const_matrix_view<T>& A,
                                                       const const_matrix_view<T>& B,
                                             T  beta, block_scatter_matrix<T,MR,NR>& C) const
{
    const T* p_a = A.data();
    const T* p_b = B.data();
          T* p_c = C.data();

    dim_t m = C.length();
    dim_t n = C.width();
    dim_t k = A.width();
    inc_t rs_c = C.row_stride();
    inc_t cs_c = C.col_stride();
    const inc_t* rscat_c = C.row_scatter();
    const inc_t* cscat_c = C.col_scatter();

    auxinfo_t data;
    //bli_auxinfo_set_next_ab(p_a, p_b, data);

    if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
    {
        ukr(k, &alpha, p_a, p_b,
            &beta, p_c, rs_c, cs_c,
            &data, nullptr);
    }
    else
    {
        T p_ab[MR*NR] __attribute__((aligned(64)));
        static constexpr T zero = 0.0;

        ukr(k, &alpha, p_a, p_b,
            &zero, p_ab, 1, MR,
            &data, nullptr);

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

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void AccumulateMicroTile<T,MR,NR>(dim_t m, dim_t n, const T* p_ab, T beta, T* p_c, inc_t rs_c, inc_t cs_c);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void AccumulateMicroTile<T,MR,NR>(dim_t m, dim_t n, const T* p_ab, T beta, T* p_c, const inc_t* rs_c, inc_t cs_c);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void AccumulateMicroTile<T,MR,NR>(dim_t m, dim_t n, const T* p_ab, T beta, T* p_c, inc_t rs_c, const inc_t* cs_c);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void AccumulateMicroTile<T,MR,NR>(dim_t m, dim_t n, const T* p_ab, T beta, T* p_c, const inc_t* rs_c, const inc_t* cs_c);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void GenericMicroKernel<T,MR,NR>(dim_t k, T alpha, const T* p_a, const T* p_b, T beta, T* p_c, inc_t rs_c, inc_t cs_c, const auxinfo_t* data);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template struct MicroKernel<MT,NT>::run<T>;
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
