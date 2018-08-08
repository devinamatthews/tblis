#ifndef _TBLIS_NODES_PACKM_HPP_
#define _TBLIS_NODES_PACKM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"

#include "memory/alignment.hpp"
#include "memory/memory_pool.hpp"

#include "matrix/normal_matrix.hpp"

#include "configs/configs.hpp"

#include "iface/1v/reduce.h"

#define TBLIS_MAX_UNROLL 8

namespace tblis
{

template <int Mat> struct pack_and_run;

template <> struct pack_and_run<matrix_constants::MAT_A>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    pack_and_run(Run& run, const communicator& comm, const config& cfg,
                 T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        A.pack(comm, cfg, false, P);
        comm.barrier();
#if 0
        if (comm.master())
        {
            printf("[%d] packing A: %.15g\n", comm.gang_num(), std::abs(reduce(single, REDUCE_NORM_1,
                row_view<const T>{{P.length(0)*P.length(1)}, P.data()}).first));

            auto ME = cfg.gemm_mr.extent<T>();

            for (len_type m = 0;m < P.length(0);m += ME)
                std::cout << matrix_view<const T>{{P.length(1), ME}, P.data() + m*P.length(1), {ME, 1}} << std::endl;

            printf("\n\n");
        }
        comm.barrier();
#endif
        run(comm, cfg, alpha, P, B, beta, C);
        comm.barrier();
    }
};

template <> struct pack_and_run<matrix_constants::MAT_B>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    pack_and_run(Run& run, const communicator& comm, const config& cfg,
                 T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        B.pack(comm, cfg, true, P);
        comm.barrier();
#if 0
        if (comm.master())
        {
            printf("[%d] packing B: %.15g\n", comm.gang_num(), std::abs(reduce(single, REDUCE_NORM_1,
                row_view<const T>{{P.length(0)*P.length(1)}, P.data()}).first));

            auto ME = cfg.gemm_nr.extent<T>();

            for (len_type m = 0;m < P.length(1);m += ME)
                std::cout << matrix_view<const T>{{P.length(0), ME}, P.data() + m*P.length(0), {ME, 1}} << std::endl;

                printf("\n\n");
        }
        comm.barrier();
#endif
        run(comm, cfg, alpha, A, P, beta, C);
        comm.barrier();
    }
};

template <int Mat, blocksize config::*BS, MemoryPool& Pool, typename Child>
struct pack
{
    Child child;
    MemoryPool::Block pack_buffer;
    void* pack_ptr = nullptr;

    pack() {}

    pack(const pack& other)
    : child(other.child) {}

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        constexpr bool Trans = (Mat == MAT_B);
        const blocksize& M = cfg.*BS;
        const len_type MR = M.def<T>();
        const len_type ME = M.extent<T>();

        len_type m_p = ceil_div(!Trans ? A.length(0) : B.length(1), MR)*ME;
        len_type k_p =         (!Trans ? A.length(1) : B.length(0));

        if (!pack_ptr)
        {
            if (comm.master())
            {
                pack_buffer = Pool.allocate<T>(m_p*k_p+std::max(m_p,k_p)*TBLIS_MAX_UNROLL);
                pack_ptr = pack_buffer.get();
            }

            comm.broadcast_value(pack_ptr);
        }

        normal_matrix<T> P(!Trans ? m_p : k_p,
                           !Trans ? k_p : m_p,
                           static_cast<T*>(pack_ptr),
                           !Trans? k_p :   1,
                           !Trans?   1 : k_p);

        pack_and_run<Mat>(child, comm, cfg, alpha, A, B, beta, C, P);
    }
};

template <MemoryPool& Pool, typename Child>
using pack_a = pack<matrix_constants::MAT_A, &config::gemm_mr, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using pack_b = pack<matrix_constants::MAT_B, &config::gemm_nr, Pool, Child>;

}

#endif
