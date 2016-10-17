#ifndef _TENSOR_TBLIS_PACK_HPP_
#define _TENSOR_TBLIS_PACK_HPP_

#include "../configs/configs.hpp.in"
#include "../memory/alignment.hpp"
#include "../memory/memory_pool.hpp"
#include "util/thread.h"
#include "util/assert.h"
#include "util/basic_types.h"
#include "util/marray.hpp"


#define TBLIS_MAX_UNROLL 8

namespace tblis
{

namespace detail
{
    extern MemoryPool BuffersForA;
    extern MemoryPool BuffersForB;
}

template <typename Config, typename T, int Mat>
struct PackRowPanel
{
    static constexpr bool Trans = Mat == matrix_constants::MAT_B;
    static constexpr len_type MR = (!Trans ? Config::template MR<T>::def
                                           : Config::template NR<T>::def);
    static constexpr len_type ME = (!Trans ? Config::template MR<T>::extent
                                           : Config::template NR<T>::extent);
    static constexpr len_type KR = Config::template KR<T>::def;
    static constexpr pack_nn_ukr_t<T> pack_nn_ukr = (!Trans ? Config::template pack_nn_mr<T>::value
                                                            : Config::template pack_nn_nr<T>::value);
    static constexpr pack_sn_ukr_t<T> pack_sn_ukr = (!Trans ? Config::template pack_sn_mr<T>::value
                                                            : Config::template pack_sn_nr<T>::value);
    static constexpr pack_ns_ukr_t<T> pack_ns_ukr = (!Trans ? Config::template pack_ns_mr<T>::value
                                                            : Config::template pack_ns_nr<T>::value);
    static constexpr pack_ss_ukr_t<T> pack_ss_ukr = (!Trans ? Config::template pack_ss_mr<T>::value
                                                            : Config::template pack_ss_nr<T>::value);
    static constexpr pack_nb_ukr_t<T> pack_nb_ukr = (!Trans ? Config::template pack_nb_mr<T>::value
                                                            : Config::template pack_nb_nr<T>::value);
    static constexpr pack_sb_ukr_t<T> pack_sb_ukr = (!Trans ? Config::template pack_sb_mr<T>::value
                                                            : Config::template pack_sb_nr<T>::value);

    void operator()(communicator& comm, matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        len_type off_first, off_last;
        std::tie(off_first, off_last, std::ignore) = comm.distribute_over_threads(m_a, MR);

        p_a += off_first*rs_a;
        p_ap += off_first*round_up(k_a, KR);

        for (len_type off_m = off_first;off_m < off_last;off_m += MR)
        {
            len_type m = std::min(MR, off_last-off_m);

            pack_nn_ukr(m, k_a, p_a, rs_a, cs_a, p_ap);

            p_a += m*rs_a;
            p_ap += ME*k_a;
        }
    }

    void operator()(thread_communicator& comm, const_scatter_matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const stride_type* rscat_a = A.scatter( Trans);
        const stride_type* cscat_a = A.scatter(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        len_type off_first, off_last;
        std::tie(off_first, off_last, std::ignore) = comm.distribute_over_threads(m_a, MR);

        p_a += off_first*rs_a;
        rscat_a += off_first;
        p_ap += off_first*round_up(k_a, KR);

        for (len_type off_m = off_first;off_m < off_last;off_m += MR)
        {
            len_type m = std::min(MR, off_last-off_m);

            if (rs_a == 0 && cs_a == 0)
            {
                pack_ss_ukr(m, k_a, p_a, rscat_a, cscat_a, p_ap);
                rscat_a += m;
            }
            else if (rs_a == 0)
            {
                pack_sn_ukr(m, k_a, p_a, rscat_a, cs_a, p_ap);
                rscat_a += m;
            }
            else if (cs_a == 0)
            {
                pack_ns_ukr(m, k_a, p_a, rs_a, cscat_a, p_ap);
                p_a += m*rs_a;
            }
            else
            {
                pack_nn_ukr(m, k_a, p_a, rs_a, cs_a, p_ap);
                p_a += m*rs_a;
            }

            p_ap += ME*k_a;
        }
    }

    void operator()(thread_communicator& comm, block_scatter_matrix<T,(Trans?KR:MR),(Trans?MR:KR)> A, matrix_view<T>& Ap) const
    {
        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        T* p_ap = Ap.data();

        len_type off_first, off_last;
        std::tie(off_first, off_last, std::ignore) = comm.distribute_over_threads(m_a, MR);

        p_ap += off_first*round_up(k_a, KR);

        A.length(Trans, MR);
        A.shift(Trans, off_first);

        for (len_type off_m = off_first;off_m < off_last;off_m += MR)
        {
            stride_type rs_a = A.stride(Trans);
            const stride_type* rscat_a = A.scatter(Trans);
            const stride_type* cscat_a = A.scatter(!Trans);
            const stride_type* cbs_a = A.block_scatter(!Trans);
            const T* p_a = A.data();
            len_type m = std::min(MR, off_last-off_m);

            if (rs_a == 0)
            {
                pack_sb_ukr(m, k_a, p_a, rscat_a, cscat_a, cbs_a, p_ap);
            }
            else
            {
                pack_nb_ukr(m, k_a, p_a, rs_a, cscat_a, cbs_a, p_ap);
            }

            p_ap += ME*k_a;
            A.shift_down(Trans);
        }

        A.shift(Trans, off_last);
        A.length(Trans, m_a);
    }
};

template <typename Pack, int Mat> struct PackAndRun;

template <typename Pack>
struct PackAndRun<Pack, matrix_constants::MAT_A>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(Run& run, const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, A, P);
        comm.barrier();
        run(cfg, comm, alpha, P, B, beta, C);
        comm.barrier();
    }
};

template <typename Pack>
struct PackAndRun<Pack, matrix_constants::MAT_B>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    PackAndRun(Run& run, const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, B, P);
        comm.barrier();
        run(cfg, comm, alpha, A, P, beta, C);
        comm.barrier();
    }
};

template <typename Config, int Mat>
struct Pack
{
    template <typename T, template <typename> class Child, template <typename> class... Children>
    struct run
    {
        typename Child<Config>::template run<T, Children...> child;

        run() {}

        run(const run& other)
        : child(other.child), pack_ptr(other.pack_ptr) {}

        MemoryPool::Block<T> pack_buffer;
        T* pack_ptr = NULL;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            constexpr bool Trans = (Mat == MAT_B);
            MemoryPool& PackBuf = (!Trans ? detail::BuffersForA
                                          : detail::BuffersForB);
            constexpr len_type MR = (!Trans ? Config::template MR<T>::def
                                            : Config::template NR<T>::def);
            constexpr len_type ME = (!Trans ? Config::template MR<T>::extent
                                            : Config::template NR<T>::extent);

            len_type m_p = ceil_div(!Trans ? A.length(0) : B.length(1), MR)*ME;
            len_type k_p =         (!Trans ? A.length(1) : B.length(0));

            if (pack_ptr == NULL)
            {
                if (comm.master())
                {
                    pack_buffer = PackBuf.allocate<T>(m_p*k_p+std::max(m_p,k_p)*TBLIS_MAX_UNROLL);
                    pack_ptr = pack_buffer;
                }

                comm.broadcast(pack_ptr);
            }

            matrix_view<T> P({!Trans ? m_p : k_p,
                              !Trans ? k_p : m_p},
                             pack_ptr,
                             {!Trans? k_p :   1,
                              !Trans?   1 : k_p});

            typedef PackRowPanel<Config, T, Mat> Pack;
            PackAndRun<Pack,Mat>(child, cfg, comm, alpha, A, B, beta, C, P);
        }
    };
};

template <typename Config>
using PackA = Pack<Config, matrix_constants::MAT_A>;

template <typename Config>
using PackB = Pack<Config, matrix_constants::MAT_B>;

}

#endif
