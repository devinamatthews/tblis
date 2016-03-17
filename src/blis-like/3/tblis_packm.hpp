#ifndef _TENSOR_TBLIS_PACK_HPP_
#define _TENSOR_TBLIS_PACK_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

namespace detail
{
    constexpr inline dim_t remainder(dim_t N, dim_t B)
    {
        return (B-1)-(N+B-1)%B;
    }

    constexpr inline dim_t round_up(dim_t N, dim_t B)
    {
        return N + remainder(N, B);
    }
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T*& p_a, inc_t rs_a, inc_t cs_a,
                    T*& p_ap);

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T*& p_a, const inc_t*& rs_a, inc_t cs_a,
                    T*& p_ap);

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T*& p_a, inc_t rs_a, const inc_t* cs_a,
                    T*& p_ap);

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T*& p_a, const inc_t*& rs_a, const inc_t* cs_a,
                    T*& p_ap);

template <typename T, dim_t MR, dim_t KR, bool Trans>
struct PackRowPanel
{
    void operator()(const Matrix<T>& A, Matrix<T>& Ap) const;

    void operator()(const ScatterMatrix<T>& A, Matrix<T>& Ap) const;

    template <bool _Trans=Trans>
    typename std::enable_if<_Trans == false>::type
    operator()(const BlockScatterMatrix<T,MR,0>& A, Matrix<T>& Ap) const;

    template <bool _Trans=Trans>
    typename std::enable_if<_Trans == true>::type
    operator()(const BlockScatterMatrix<T,0,MR>& A, Matrix<T>& Ap) const;
};

template <typename T>
struct PackNoop
{
    void operator()(Matrix<T>& A, Matrix<T>& Ap) const
    {
        memset((T*)Ap, 0, Ap.width()*Ap.length()*sizeof(T));
    }

    void operator()(ScatterMatrix<T>& A, Matrix<T>& Ap) const
    {
        memset((T*)Ap, 0, Ap.width()*Ap.length()*sizeof(T));
    }

    template <dim_t UB, dim_t VB>
    void operator()(BlockScatterMatrix<T,UB,VB>& A, Matrix<T>& Ap) const
    {
        memset((T*)Ap, 0, Ap.width()*Ap.length()*sizeof(T));
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

            constexpr dim_t MR = MT<T>::def;
            constexpr dim_t KR = KT<T>::def;

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
}

#endif
