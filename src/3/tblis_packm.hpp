#ifndef _TENSOR_TBLIS_PACK_HPP_
#define _TENSOR_TBLIS_PACK_HPP_

#include "tblis.hpp"

#define TBLIS_MAX_UNROLL 8

namespace tblis
{

namespace detail
{
    extern MemoryPool BuffersForA;
    extern MemoryPool BuffersForB;
}

template <typename T, idx_type MR, idx_type KR>
void PackMicroPanel(idx_type m, idx_type k,
                    const T* restrict & p_a, stride_type rs_a, stride_type cs_a,
                    T* restrict & p_ap)
{
    idx_type k_rem = remainder(k, KR);

    if (m == MR)
    {
        if (rs_a == 1)
        {
            for (idx_type i = 0;i < k;i++)
            {
                for (idx_type mr = 0;mr < MR;mr++)
                {
                    p_ap[mr] = p_a[mr];
                }

                p_a += cs_a;
                p_ap += MR;
            }
        }
        else if (cs_a == 1)
        {
            for (idx_type mr = 0;mr < MR;mr++)
            {
                for (idx_type i = 0;i < k;i++)
                {
                    p_ap[MR*i+mr] = p_a[i+rs_a*mr];
                }
            }

            p_a += cs_a*k;
            p_ap += MR*k;
        }
        else
        {
            for (idx_type i = 0;i < k;i++)
            {
                for (idx_type mr = 0;mr < MR;mr++)
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
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[rs_a*mr];
            }

            for (idx_type mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }

    for (idx_type i = 0;i < k_rem;i++)
    {
        for (idx_type mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a += rs_a*m - cs_a*k;
}

template <typename T, idx_type MR, idx_type KR>
void PackMicroPanel(idx_type m, idx_type k,
                    const T* restrict & p_a,
                    const stride_type* restrict & rs_a, stride_type cs_a,
                    T* restrict & p_ap)
{
    idx_type k_rem = remainder(k, KR);

    if (m == MR)
    {
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[rs_a[mr]];
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }
    else
    {
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[rs_a[mr]];
            }

            for (idx_type mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_a += cs_a;
            p_ap += MR;
        }
    }

    for (idx_type i = 0;i < k_rem;i++)
    {
        for (idx_type mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a -= cs_a*k;
    rs_a += m;
}

template <typename T, idx_type MR, idx_type KR>
void PackMicroPanel(idx_type m, idx_type k,
                    const T* restrict & p_a,
                    stride_type rs_a, const stride_type* restrict cs_a,
                    T* restrict & p_ap)
{
    idx_type k_rem = remainder(k, KR);

    if (m == MR)
    {
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a*mr];
            }

            p_ap += MR;
        }
    }
    else
    {
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a*mr];
            }

            for (idx_type mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_ap += MR;
        }
    }

    for (idx_type i = 0;i < k_rem;i++)
    {
        for (idx_type mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    p_a += rs_a*m;
}

template <typename T, idx_type MR, idx_type KR>
void PackMicroPanel(idx_type m, idx_type k,
                    const T* restrict & p_a,
                    const stride_type* restrict & rs_a, const stride_type* restrict cs_a,
                    T* restrict & p_ap)
{
    idx_type k_rem = remainder(k, KR);

    if (m == MR)
    {
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < MR;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a[mr]];
            }

            p_ap += MR;
        }
    }
    else
    {
        for (idx_type i = 0;i < k;i++)
        {
            for (idx_type mr = 0;mr < m;mr++)
            {
                p_ap[mr] = p_a[cs_a[i] + rs_a[mr]];
            }

            for (idx_type mr = m;mr < MR;mr++)
            {
                p_ap[mr] = T();
            }

            p_ap += MR;
        }
    }

    for (idx_type i = 0;i < k_rem;i++)
    {
        for (idx_type mr = 0;mr < MR;mr++)
        {
            p_ap[mr] = T();
        }

        p_ap += MR;
    }

    rs_a += m;
}

template <typename T, idx_type MR, idx_type KR, bool Trans>
struct PackRowPanel
{
    void operator()(ThreadCommunicator& comm, matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        //printf("before: %.15f\n", (double)real(tblis_normfm(A)));

        idx_type m_a = A.length( Trans);
        idx_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        idx_type off_first, off_last;
        std::tie(off_first, off_last, std::ignore) = comm.distribute_over_threads(m_a, MR);
        //printf_locked("pack [%d/%d] %d %d\n", comm.thread_num(), comm.num_threads(), off_first, off_last);

        p_a += off_first*rs_a;
        p_ap += off_first*round_up(k_a, KR);

        for (idx_type off_m = off_first;off_m < off_last;off_m += MR)
        {
            //printf("%d %d : %d %ld\n", off_m, std::min(MR, off_last-off_m), MR, p_ap-Ap.data());
            //printf("bsub: %.15f\n", (double)real(tblis_normfm(std::min(MR, off_last-off_m), k_a, p_a, rs_a, cs_a)));
            //auto p_ap_old = p_ap;
            PackMicroPanel<T,MR,KR>(std::min(MR, off_last-off_m), k_a,
                                    p_a, rs_a, cs_a, p_ap);
            //printf("asub: %.15f\n", (double)real(tblis_normfm(MR, k_a, p_ap_old, 1, MR)));
        }

        //comm.barrier();
        //printf_locked("%s[%d]: %f\n", (Trans ? "B" : "A"), comm.thread_num(), pow((double)real(tblis_normfm(Ap)),2));
        //printf("%d %d %ld %ld\n", Ap.length(0), Ap.length(1), Ap.stride(0), Ap.stride(1));
        //printf("after: %.15f\n", (double)real(tblis_normfm(Ap)));
    }

    void operator()(ThreadCommunicator& comm, const_scatter_matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        idx_type m_a = A.length( Trans);
        idx_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const stride_type* rscat_a = A.scatter( Trans);
        const stride_type* cscat_a = A.scatter(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        idx_type off_first, off_last;
        std::tie(off_first, off_last, std::ignore) = comm.distribute_over_threads(m_a, MR);

        p_a += off_first*rs_a;
        rscat_a += off_first;
        p_ap += off_first*round_up(k_a, KR);

        for (idx_type off_m = off_first;off_m < off_last;off_m += MR)
        {
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
        }
    }

    void operator()(ThreadCommunicator& comm, block_scatter_matrix<T,(Trans?KR:MR),(Trans?MR:KR)> A, matrix_view<T>& Ap) const
    {
        idx_type m_a = A.length( Trans);
        idx_type k_a = A.length(!Trans);
        T* p_ap = Ap.data();

        idx_type off_first, off_last;
        std::tie(off_first, off_last, std::ignore) = comm.distribute_over_threads(m_a, MR);

        p_ap += off_first*round_up(k_a, KR);

        A.length(Trans, MR);
        A.shift_down(Trans, off_first);

        for (idx_type off_m = off_first;off_m < off_last;off_m += MR)
        {
            stride_type rs_a = A.stride( Trans);
            stride_type cs_a = A.stride(!Trans);
            const stride_type* rscat_a = A.scatter( Trans);
            const stride_type* cscat_a = A.scatter(!Trans);
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

            A.shift_down(Trans);
        }

        A.shift_up(Trans, off_last);
        A.length(Trans, m_a);
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

            constexpr idx_type MR = (Trans ? NT<T>::def : MT<T>::def);
            constexpr idx_type NR = (Trans ? MT<T>::def : NT<T>::def);

            idx_type m_p = (Trans ? B.length(1) : A.length(0));
            idx_type n_p = (Trans ? B.length(0) : A.length(1));
            m_p = round_up(m_p, MR);
            n_p = round_up(n_p, NR);

            if (pack_ptr == NULL)
            {
                if (comm.master())
                {
                    pack_buffer = PackBuf.allocate<T>(m_p*n_p+std::max(m_p,n_p)*TBLIS_MAX_UNROLL);
                    pack_ptr = pack_buffer;
                }

                comm.broadcast(pack_ptr);
            }

            //printf_locked("pack_ptr %d [%d/%d:%d] %p\n", Mat,
            //              comm.thread_num(), comm.num_threads(),
            //              comm.gang_num(), pack_ptr);

            matrix_view<T> P(Trans ? n_p : m_p,
                             Trans ? m_p : n_p,
                             pack_ptr,
                             Trans?   1 : n_p,
                             Trans? n_p :   1);

            assert(P.length(0) == Trans ? n_p : m_p);
            assert(P.length(1) == Trans ? m_p : n_p);
            assert(P.data() == pack_ptr);
            assert(P.stride(0) == Trans ?   1 : n_p);
            assert(P.stride(1) == Trans ? n_p :   1);

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

#endif
