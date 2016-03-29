#ifndef _TENSOR_TBLIS_PACK_HPP_
#define _TENSOR_TBLIS_PACK_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

namespace detail
{
    extern MemoryPool BuffersForA;
    extern MemoryPool BuffersForB;

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
    void operator()(ThreadCommunicator& comm, const Matrix<T>& A, Matrix<T>& Ap) const;

    void operator()(ThreadCommunicator& comm, const ScatterMatrix<T>& A, Matrix<T>& Ap) const;

    void operator()(ThreadCommunicator& comm, const BlockScatterMatrix<T,(Trans ? 0 : MR),(Trans ? MR : 0)>& A, Matrix<T>& Ap) const;
};

template <typename T>
struct PackNoop
{
    void operator()(ThreadCommunicator& comm, Matrix<T>& A, Matrix<T>& Ap) const
    {
        memset((T*)Ap, 0, Ap.width()*Ap.length()*sizeof(T));
    }

    void operator()(ThreadCommunicator& comm, ScatterMatrix<T>& A, Matrix<T>& Ap) const
    {
        memset((T*)Ap, 0, Ap.width()*Ap.length()*sizeof(T));
    }

    template <dim_t UB, dim_t VB>
    void operator()(ThreadCommunicator& comm, BlockScatterMatrix<T,UB,VB>& A, Matrix<T>& Ap) const
    {
        memset((T*)Ap, 0, Ap.width()*Ap.length()*sizeof(T));
    }
};

template <typename Pack, typename Run, int Mat>
struct PackAndRun
{
    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(Run& run, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, A, P);
        comm.barrier();
        run(comm, alpha, P, B, beta, C);
        comm.barrier();
    }
};

template <typename Pack, typename Run>
struct PackAndRun<Pack, Run, matrix_constants::MAT_B>
{
    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(Run& run, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, B, P);
        comm.barrier();
        run(comm, alpha, A, P, beta, C);
        comm.barrier();
    }
};

template <template <typename> class MT, template <typename> class KT, int Mat>
struct Pack
{
    template <typename T, typename Child, typename... Children>
    struct run
    {
        typename Child::template run<T, Children...> child;

        T* pack_buffer = NULL;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            constexpr dim_t MR = MT<T>::def;
            constexpr dim_t KR = KT<T>::def;

            MemoryPool& PackBuf = (Mat == MAT_A ? detail::BuffersForA
                                                : detail::BuffersForB);
            constexpr bool Trans = (Mat == MAT_B);

            typedef PackRowPanel<T,MR,KR,Trans> Pack;
            typedef typename Child::template run<T, Children...> Run;

            dim_t m_p = (Mat == MAT_A ? A.length() : B.width ());
            dim_t k_p = (Mat == MAT_A ? A.width () : B.length());
            m_p = detail::round_up(m_p, MR);
            k_p = detail::round_up(k_p, KR);
            MemoryPool::Block<T> buf;
            T* ptr;

            if (comm.thread_num() == 0)
            {
                buf = PackBuf.allocate<T>(m_p*k_p + extra_space);
                ptr = buf;
            }

            comm.broadcast(ptr);

            Matrix<T> P((Mat == MAT_A ? m_p : k_p),
                        (Mat == MAT_A ? k_p : m_p),
                        pack_buffer,
                        (Mat == MAT_A ? k_p :   1),
                        (Mat == MAT_A ?   1 : k_p));

            PackAndRun<Pack,Run,Mat>(child, comm, alpha, A, B, beta, C, P);
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
