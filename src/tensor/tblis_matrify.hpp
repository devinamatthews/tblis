#ifndef _TBLIS_MATRIFY_HPP_
#define _TBLIS_MATRIFY_HPP_

#include "tblis_config.hpp"
#include "tblis_configs.hpp"
#include "tblis_marray.hpp"
#include "tblis_packm.hpp"
#include "tblis_tensor_matrix.hpp"

namespace tblis
{

namespace detail
{
    extern MemoryPool BuffersForScatter;
}

template <typename T, len_type MR, len_type NR>
void BlockScatter(thread_communicator& comm, tensor_matrix<T>& A, stride_type* rs, stride_type* cs, stride_type* rscat, stride_type* cscat)
{
    len_type m = A.length(0);
    len_type n = A.length(1);

    len_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(m, MR);

    A.length(0, last-first);
    A.shift(0, first);
    A.template fill_block_scatter<MR>(0, rs+first/MR, rscat+first);
    A.shift(0, -first);
    A.length(0, m);

    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n, NR);

    A.length(1, last-first);
    A.shift(1, first);
    A.template fill_block_scatter<NR>(1, cs+first/NR, cscat+first);
    A.shift(1, -first);
    A.length(1, n);

    comm.barrier();
}

template <len_type MR, len_type NR, int Mat> struct MatrifyAndRun;

template <len_type MR, len_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_A>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        //BlockScatter<T,MR,NR>(comm, A, parent.rs, parent.cs, parent.rscat, parent.cscat);
        A.template fill_block_scatter<MR>(0, parent.rs, parent.rscat);
        A.template fill_block_scatter<NR>(1, parent.cs, parent.cscat);
        block_scatter_matrix<T,MR,NR> M(A.length(0), A.length(1), A.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(A.length(0), A.length(1), A.data(), parent.rscat, parent.cscat);
        parent.child(cfg, comm, alpha, M, B, beta, C);
    }
};

template <len_type MR, len_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_B>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        //BlockScatter<T,MR,NR>(comm, B, parent.rs, parent.cs, parent.rscat, parent.cscat);
        B.template fill_block_scatter<MR>(0, parent.rs, parent.rscat);
        B.template fill_block_scatter<NR>(1, parent.cs, parent.cscat);
        block_scatter_matrix<T,MR,NR> M(B.length(0), B.length(1), B.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(B.length(0), B.length(1), B.data(), parent.rscat, parent.cscat);
        parent.child(cfg, comm, alpha, A, M, beta, C);
    }
};

template <len_type MR, len_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_C>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        //BlockScatter<T,MR,NR>(comm, C, parent.rs, parent.cs, parent.rscat, parent.cscat);
        C.template fill_block_scatter<MR>(0, parent.rs, parent.rscat);
        C.template fill_block_scatter<NR>(1, parent.cs, parent.cscat);
        block_scatter_matrix<T,MR,NR> M(C.length(0), C.length(1), C.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(C.length(0), C.length(1), C.data(), parent.rscat, parent.cscat);
        parent.child(cfg, comm, alpha, A, B, beta, M);
    }
};

template <typename Config, int DimM, int DimN, int Mat>
struct Matrify
{
    template <typename T, template <typename> class Child, template <typename> class... Children>
    struct run
    {
        typename Child<Config>::template run<T, Children...> child;

        run() {}

        run(const run& other)
        : child(other.child), rscat(other.rscat), cscat(other.cscat),
          rs(other.rs), cs(other.cs) {}

        MemoryPool::Block<stride_type> scat_buffer;
        stride_type* rscat = NULL;
        stride_type* cscat = NULL;
        stride_type* rs = NULL;
        stride_type* cs = NULL;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            using MB = typename config_traits<Config>::template BS<T,DimM>;
            using NB = typename config_traits<Config>::template BS<T,DimN>;
            constexpr len_type MR = MB::def;
            constexpr len_type NR = NB::def;

            len_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
            len_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));

            if (rscat == NULL)
            {
                if (comm.master())
                {
                    scat_buffer = detail::BuffersForScatter.allocate<stride_type>(2*m + 2*n);
                    rscat = scat_buffer;
                }

                comm.broadcast(rscat);

                cscat = rscat+m;
                rs = cscat+n;
                cs = rs+m;
            }

            MatrifyAndRun<MR,NR,Mat>(*this, cfg, comm, alpha, A, B, beta, C);
        }
    };
};

template <typename Config>
using MatrifyA = Matrify<Config, matrix_constants::DIM_MR, matrix_constants::DIM_KR, matrix_constants::MAT_A>;

template <typename Config>
using MatrifyB = Matrify<Config, matrix_constants::DIM_KR, matrix_constants::DIM_NR, matrix_constants::MAT_B>;

template <typename Config>
using MatrifyC = Matrify<Config, matrix_constants::DIM_MR, matrix_constants::DIM_NR, matrix_constants::MAT_C>;

template <typename Config, int DimM, int DimN, int Mat>
struct MatrifyAndPack
{
    template <typename T, template <typename> class... Children>
    struct run : Matrify<Config, DimM, DimN, Mat>::template run<T, Pack<Config, DimM, DimN, Mat>, Children...>
    {
        typedef typename Matrify<Config, DimM, DimN, Mat>::template run<T, Pack<Config, DimM, DimN, Mat>, Children...> Sib;

        using Sib::child;
        using Sib::rscat;
        using Sib::cscat;
        using Sib::rs;
        using Sib::cs;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            using MB = typename config_traits<Config>::template BS<T,DimM>;
            using NB = typename config_traits<Config>::template BS<T,DimN>;
            constexpr len_type MR = MB::def;
            constexpr len_type NR = NB::def;

            len_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
            len_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));
            m = round_up(m, MR);
            n = round_up(n, NR);

            MemoryPool& PackBuf = (Mat == MAT_A ? detail::BuffersForA : detail::BuffersForB);

            auto& pack_buffer = child.pack_buffer;
            T*& pack_ptr = child.pack_ptr;

            if (pack_ptr == NULL)
            {
                if (comm.master())
                {
                    len_type scatter_size = size_as_type<stride_type,T>(2*m + 2*n);
                    pack_buffer = PackBuf.allocate<T>(m*n + std::max(m,n)*TBLIS_MAX_UNROLL + scatter_size);
                    pack_ptr = pack_buffer;
                }

                comm.broadcast(pack_ptr);

                rscat = convert_and_align<T,stride_type>(pack_ptr + m*n);
                cscat = rscat+m;
                rs = cscat+n;
                cs = rs+m;
            }

            Sib::operator()(cfg, comm, alpha, A, B, beta, C);
        }
    };
};

template <typename Config>
using MatrifyAndPackA = MatrifyAndPack<Config, matrix_constants::DIM_MR, matrix_constants::DIM_KR, matrix_constants::MAT_A>;

template <typename Config>
using MatrifyAndPackB = MatrifyAndPack<Config, matrix_constants::DIM_KR, matrix_constants::DIM_NR, matrix_constants::MAT_B>;

}

#endif
