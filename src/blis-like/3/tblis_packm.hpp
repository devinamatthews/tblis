#ifndef _TENSOR_TBLIS_PACK_HPP_
#define _TENSOR_TBLIS_PACK_HPP_

#include "tblis.hpp"

#define TBLIS_MAX_UNROLL 8

namespace tblis
{
namespace blis_like
{

namespace detail
{
    extern MemoryPool BuffersForA;
    extern MemoryPool BuffersForB;
}

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a, inc_t rs_a, inc_t cs_a,
                    T* restrict & p_ap);

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a, const inc_t* restrict & rs_a, inc_t cs_a,
                    T* restrict & p_ap);

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a, inc_t rs_a, const inc_t* restrict  cs_a,
                    T* restrict & p_ap);

template <typename T, dim_t MR, dim_t KR>
void PackMicroPanel(dim_t m, dim_t k,
                    const T* restrict & p_a, const inc_t* restrict & rs_a, const inc_t* restrict  cs_a,
                    T* restrict & p_ap);

template <typename T, dim_t MR, dim_t KR, bool Trans>
struct PackRowPanel
{
    void operator()(ThreadCommunicator& comm, const Matrix<T>& A, Matrix<T>& Ap) const;

    void operator()(ThreadCommunicator& comm, const ScatterMatrix<T>& A, Matrix<T>& Ap) const;

    void operator()(ThreadCommunicator& comm, const BlockScatterMatrix<T,(Trans ? KR : MR),(Trans ? MR : KR)>& A, Matrix<T>& Ap) const;
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

template <typename Pack, int Mat> struct PackAndRun;

template <typename Pack>
struct PackAndRun<Pack, matrix_constants::MAT_A>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(Run& run, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        comm.barrier();
        Pack()(comm, A, P);
        comm.barrier();
        run(comm, alpha, P, B, beta, C);
    }
};

template <typename Pack>
struct PackAndRun<Pack, matrix_constants::MAT_B>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(Run& run, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        comm.barrier();
        Pack()(comm, B, P);
        comm.barrier();
        run(comm, alpha, A, P, beta, C);
    }
};

template <template <typename> class MT, template <typename> class NT, int Mat>
struct Pack
{
    template <typename T, typename Child, typename... Children>
    struct run
    {
        typename Child::template run<T, Children...> child;

        run() {}

        run(const run& other)
        : child(other.child), pack_ptr(other.pack_ptr) {}

        MemoryPool::Block<T> pack_buffer;
        T* pack_ptr = NULL;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            MemoryPool& PackBuf = (Mat == MAT_A ? detail::BuffersForA
                                                : detail::BuffersForB);
            constexpr bool Trans = (Mat == MAT_B);

            constexpr dim_t MR = (Trans ? NT<T>::def : MT<T>::def);
            constexpr dim_t NR = (Trans ? MT<T>::def : NT<T>::def);

            dim_t m_p = (Mat == MAT_A ? A.length() : B.width ());
            dim_t n_p = (Mat == MAT_A ? A.width () : B.length());
            m_p = util::round_up(m_p, MR);
            n_p = util::round_up(n_p, NR);

            if (pack_ptr == NULL)
            {
                if (comm.master())
                {
                    pack_buffer = PackBuf.allocate<T>(m_p*(n_p+TBLIS_MAX_UNROLL));
                    pack_ptr = pack_buffer;
                }

                comm.broadcast(pack_ptr);
            }

            Matrix<T> P((Mat == MAT_A ? m_p : n_p),
                        (Mat == MAT_A ? n_p : m_p),
                        pack_ptr,
                        (Mat == MAT_A ? n_p :   1),
                        (Mat == MAT_A ?   1 : n_p));

            typedef PackRowPanel<T,MR,NR,Trans> Pack;
            PackAndRun<Pack,Mat>(child, comm, alpha, A, B, beta, C, P);
        }
    };
};

template <template <typename> class MT, template <typename> class KT>
using PackA = Pack<MT,KT,matrix_constants::MAT_A>;

template <template <typename> class KT, template <typename> class NT>
using PackB = Pack<KT,NT,matrix_constants::MAT_B>;

}
}

#endif
