#ifndef _TBLIS_NODES_MATRIFY_HPP_
#define _TBLIS_NODES_MATRIFY_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "matrix/tensor_matrix.hpp"

#include "nodes/packm.hpp"

#include "configs/configs.hpp"

namespace tblis
{

namespace detail
{
    extern MemoryPool BuffersForScatter;
}

template <typename MatrixA>
void block_scatter(const communicator& comm, MatrixA& A,
                   stride_type* rscat, len_type MB, stride_type* rbs,
                   stride_type* cscat, len_type NB, stride_type* cbs)
{
    len_type m = A.length(0);
    len_type n = A.length(1);

    len_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(m, MB);

    A.length(0, last-first);
    A.shift(0, first);
    A.fill_block_scatter(0, rscat+first, MB, rbs+first/MB);
    A.shift(0, -first);
    A.length(0, m);

    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n, NB);

    A.length(1, last-first);
    A.shift(1, first);
    A.fill_block_scatter(1, cscat+first, NB, cbs+first/NB);
    A.shift(1, -first);
    A.length(1, n);

    comm.barrier();
}

template <int Mat> struct matrify_and_run;

template <> struct matrify_and_run<matrix_constants::MAT_A>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    matrify_and_run(Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        const len_type MB = cfg.gemm_mr.def<T>();
        const len_type NB = cfg.gemm_kr.def<T>();

        //block_scatter(comm, A, parent.rscat, MB, parent.rbs,
        //                       parent.cscat, NB, parent.cbs);

        A.fill_block_scatter(0, parent.rscat, MB, parent.rbs);
        A.fill_block_scatter(1, parent.cscat, NB, parent.cbs);

        block_scatter_matrix<T> M(A.length(0), A.length(1), A.data(),
                                  parent.rscat, MB, parent.rbs,
                                  parent.cscat, NB, parent.cbs);

        parent.child(comm, cfg, alpha, M, B, beta, C);
    }
};

template <> struct matrify_and_run<matrix_constants::MAT_B>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    matrify_and_run(Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        const len_type MB = cfg.gemm_kr.def<T>();
        const len_type NB = cfg.gemm_nr.def<T>();

        //block_scatter(comm, B, parent.rscat, MB, parent.rbs,
        //                       parent.cscat, NB, parent.cbs);

        B.fill_block_scatter(0, parent.rscat, MB, parent.rbs);
        B.fill_block_scatter(1, parent.cscat, NB, parent.cbs);

        block_scatter_matrix<T> M(B.length(0), B.length(1), B.data(),
                                  parent.rscat, MB, parent.rbs,
                                  parent.cscat, NB, parent.cbs);

        parent.child(comm, cfg, alpha, A, M, beta, C);
    }
};

template <> struct matrify_and_run<matrix_constants::MAT_C>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    matrify_and_run(Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        const len_type MB = cfg.gemm_mr.def<T>();
        const len_type NB = cfg.gemm_nr.def<T>();

        //block_scatter(comm, C, parent.rscat, MB, parent.rbs,
        //                       parent.cscat, NB, parent.cbs);

        C.fill_block_scatter(0, parent.rscat, MB, parent.rbs);
        C.fill_block_scatter(1, parent.cscat, NB, parent.cbs);

        block_scatter_matrix<T> M(C.length(0), C.length(1), C.data(),
                                  parent.rscat, MB, parent.rbs,
                                  parent.cscat, NB, parent.cbs);

        parent.child(comm, cfg, alpha, A, B, beta, M);
    }
};

template <int Mat, MemoryPool& Pool, typename Child>
struct matrify
{
    Child child;
    MemoryPool::Block scat_buffer;
    stride_type* rscat = nullptr;
    stride_type* cscat = nullptr;
    stride_type* rbs = nullptr;
    stride_type* cbs = nullptr;

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        len_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
        len_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));

        if (!rscat)
        {
            if (comm.master())
            {
                scat_buffer = Pool.allocate<stride_type>(2*m + 2*n);
                rscat = scat_buffer.get<stride_type>();
            }

            comm.broadcast(rscat);

            cscat = rscat+m;
            rbs = cscat+n;
            cbs = rbs+m;
        }

        matrify_and_run<Mat>(*this, comm, cfg, alpha, A, B, beta, C);
    }
};

template <MemoryPool& Pool, typename Child>
using matrify_a = matrify<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using matrify_b = matrify<matrix_constants::MAT_B, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using matrify_c = matrify<matrix_constants::MAT_C, Pool, Child>;

template <int Mat, MemoryPool& Pool, typename Child>
struct matrify_and_pack : matrify<Mat, Pool, pack<Mat, Pool, Child>>
{
    typedef matrify<Mat, Pool, pack<Mat, Pool, Child>> Sib;

    using Sib::child;
    using Sib::rscat;
    using Sib::cscat;
    using Sib::rbs;
    using Sib::cbs;

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        const len_type MR = (Mat == MAT_B ? cfg.gemm_kr.def<T>()
                                          : cfg.gemm_mr.def<T>());
        const len_type NR = (Mat == MAT_A ? cfg.gemm_kr.def<T>()
                                          : cfg.gemm_nr.def<T>());

        len_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
        len_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));
        m = round_up(m, MR);
        n = round_up(n, NR);

        auto& pack_buffer = child.pack_buffer;
        auto& pack_ptr = child.pack_ptr;

        if (!pack_ptr)
        {
            if (comm.master())
            {
                len_type scatter_size = size_as_type<stride_type,T>(2*m + 2*n);
                pack_buffer = Pool.allocate<T>(m*n + std::max(m,n)*TBLIS_MAX_UNROLL + scatter_size);
                pack_ptr = pack_buffer.get();
            }

            comm.broadcast(pack_ptr);

            rscat = convert_and_align<T,stride_type>(static_cast<T*>(pack_ptr) + m*n);
            cscat = rscat+m;
            rbs = cscat+n;
            cbs = rbs+m;
        }

        Sib::operator()(comm, cfg, alpha, A, B, beta, C);
    }
};

template <MemoryPool& Pool, typename Child>
using matrify_and_pack_a = matrify_and_pack<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using matrify_and_pack_b = matrify_and_pack<matrix_constants::MAT_B, Pool, Child>;

}

#endif
