#ifndef _TBLIS_NODES_MATRIFY_HPP_
#define _TBLIS_NODES_MATRIFY_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "matrix/block_scatter_matrix.hpp"
#include "matrix/patch_block_scatter_matrix.hpp"

#include "nodes/packm.hpp"

#include "configs/configs.hpp"

namespace tblis
{

namespace detail
{

extern MemoryPool BuffersForScatter;

template <typename Step>
struct is_pack : std::false_type {};

template <int Mat, blocksize config::*BS, MemoryPool& Pool, typename Child>
struct is_pack<pack<Mat, BS, Pool, Child>> : std::true_type {};

}

template <int Mat> struct matrify_and_run;

template <int Mat, blocksize config::*MBS, blocksize config::*NBS, MemoryPool& Pool, typename Child>
struct matrify;

template <typename Matrify, typename Child, typename MatrixA>
detail::enable_if_t<!detail::is_pack<Child>::value>
allocate_buffers(len_type MB, len_type NB, Matrify& parent, Child&,
                 const communicator& comm, MatrixA& A)
{
    if (!parent.rscat)
    {
        unsigned mp = A.num_patches(0);
        unsigned np = A.num_patches(1);
        len_type m = A.length(0) + (MB-1)*mp;
        len_type n = A.length(1) + (NB-1)*np;

        if (comm.master())
        {
            len_type patch_size = size_as_type<block_scatter_matrix<float>,stride_type>(mp*np);
            parent.scat_buffer = parent.pool.template allocate<stride_type>(2*m*np + 2*n*mp + patch_size);
            parent.rscat = parent.scat_buffer.template get<stride_type>();
        }

        comm.broadcast_value(parent.rscat);

        parent.cscat = parent.rscat+m*np;
        parent.rbs = parent.cscat+n*mp;
        parent.cbs = parent.rbs+m*np;
        parent.patches = convert_and_align<stride_type,block_scatter_matrix<float>>(parent.cbs+n*mp);
    }
}

template <typename Matrify, typename Child, typename MatrixA>
detail::enable_if_t<detail::is_pack<Child>::value>
allocate_buffers(len_type MB, len_type NB, Matrify& parent, Child& child,
                 const communicator& comm, MatrixA& A)
{
    typedef typename MatrixA::value_type T;

    if (!parent.rscat)
    {
        unsigned mp = A.num_patches(0);
        unsigned np = A.num_patches(1);
        len_type m = A.length(0) + (MB-1)*mp;
        len_type n = A.length(1) + (NB-1)*np;

        if (comm.master())
        {
            len_type scatter_size = size_as_type<stride_type,T>(2*m*np + 2*n*mp) +
                                    size_as_type<block_scatter_matrix<float>,T>(mp*np);
            child.pack_buffer = parent.pool.template allocate<T>(m*n + std::max(m,n)*TBLIS_MAX_UNROLL + scatter_size);
            child.pack_ptr = child.pack_buffer.get();
        }

        comm.broadcast_value(child.pack_ptr);

        parent.rscat = convert_and_align<T,stride_type>(static_cast<T*>(child.pack_ptr) + m*n);
        parent.cscat = parent.rscat+m*np;
        parent.rbs = parent.cscat+n*mp;
        parent.cbs = parent.rbs+m*np;
        parent.patches = convert_and_align<stride_type,block_scatter_matrix<float>>(parent.cbs+n*mp);
    }
}

template <> struct matrify_and_run<matrix_constants::MAT_A>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    matrify_and_run(len_type MB, len_type NB, Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        allocate_buffers(MB, NB, parent, parent.child, comm, A);
        patch_block_scatter_matrix<T> M(comm, A,
                                        MB, MB, parent.rscat, parent.rbs,
                                        NB,  1, parent.cscat, parent.cbs,
                                        static_cast<block_scatter_matrix<T>*>(parent.patches));
        parent.child(comm, cfg, alpha, M, B, beta, C);
    }
};

template <> struct matrify_and_run<matrix_constants::MAT_B>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    matrify_and_run(len_type MB, len_type NB, Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        allocate_buffers(MB, NB, parent, parent.child, comm, B);
        patch_block_scatter_matrix<T> M(comm, B,
                                        MB,  1, parent.rscat, parent.rbs,
                                        NB, NB, parent.cscat, parent.cbs,
                                        static_cast<block_scatter_matrix<T>*>(parent.patches));
        parent.child(comm, cfg, alpha, A, M, beta, C);
    }
};

template <> struct matrify_and_run<matrix_constants::MAT_C>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    matrify_and_run(len_type MB, len_type NB, Parent& parent, const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        allocate_buffers(MB, NB, parent, parent.child, comm, C);
        patch_block_scatter_matrix<T> M(comm, C,
                                        MB, MB, parent.rscat, parent.rbs,
                                        NB, NB, parent.cscat, parent.cbs,
                                        static_cast<block_scatter_matrix<T>*>(parent.patches));
        parent.child(comm, cfg, alpha, A, B, beta, M);
    }
};

template <int Mat, blocksize config::*MBS, blocksize config::*NBS, MemoryPool& Pool, typename Child>
struct matrify
{
    static constexpr MemoryPool& pool = Pool;
    Child child;
    MemoryPool::Block scat_buffer;
    stride_type* rscat = nullptr;
    stride_type* cscat = nullptr;
    stride_type* rbs = nullptr;
    stride_type* cbs = nullptr;
    void* patches = nullptr;

    matrify() {}

    matrify(const matrify& other)
    : child(other.child) {}

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        auto MB = (cfg.*MBS).def<T>();
        auto NB = (cfg.*NBS).def<T>();

        if (Mat == matrix_constants::MAT_B) std::swap(MB, NB);

        matrify_and_run<Mat>(MB, NB, *this, comm, cfg, alpha, A, B, beta, C);
    }
};

template <MemoryPool& Pool, typename Child>
using matrify_a = matrify<matrix_constants::MAT_A, &config::gemm_mr, &config::gemm_kr, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using matrify_b = matrify<matrix_constants::MAT_B, &config::gemm_nr, &config::gemm_kr, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using matrify_c = matrify<matrix_constants::MAT_C, &config::gemm_mr, &config::gemm_nr, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using matrify_and_pack_a = matrify<matrix_constants::MAT_A, &config::gemm_mr, &config::gemm_kr, Pool,
                             pack<matrix_constants::MAT_A, &config::gemm_mr, Pool, Child>>;

template <MemoryPool& Pool, typename Child>
using matrify_and_pack_b = matrify<matrix_constants::MAT_B, &config::gemm_nr, &config::gemm_kr, Pool,
                             pack<matrix_constants::MAT_B, &config::gemm_nr, Pool, Child>>;

}

#endif
