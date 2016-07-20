#ifndef _TBLIS_MATRIFY_HPP_
#define _TBLIS_MATRIFY_HPP_

#include "tblis.hpp"

namespace tblis
{

namespace detail
{
    extern MemoryPool BuffersForScatter;
}

template <typename T, idx_type MR, idx_type NR>
void BlockScatter(thread_communicator& comm, tensor_matrix<T>& A, stride_type* rs, stride_type* cs, stride_type* rscat, stride_type* cscat)
{
    idx_type m = A.length(0);
    idx_type n = A.length(1);

    idx_type first, last;
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

template <idx_type MR, idx_type NR, int Mat> struct MatrifyAndRun;

template <idx_type MR, idx_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_A>
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

template <idx_type MR, idx_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_B>
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

template <idx_type MR, idx_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_C>
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

template <template <typename> class MT, template <typename> class NT, int Mat>
struct Matrify
{
    template <typename T, typename Child, typename... Children>
    struct run
    {
        typename Child::template run<T, Children...> child;

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

            constexpr idx_type MR = MT<T>::def;
            constexpr idx_type NR = NT<T>::def;

            idx_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
            idx_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));

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

template <template <typename> class MR, template <typename> class KR>
using MatrifyA = Matrify<MR,KR,matrix_constants::MAT_A>;

template <template <typename> class KR, template <typename> class NR>
using MatrifyB = Matrify<KR,NR,matrix_constants::MAT_B>;

template <template <typename> class MR, template <typename> class NR>
using MatrifyC = Matrify<MR,NR,matrix_constants::MAT_C>;

template <template <typename> class MT, template <typename> class NT, int Mat>
struct MatrifyAndPack
{
    template <typename T, typename... Children>
    struct run : Matrify<MT, NT, Mat>::template run<T, Pack<MT, NT, Mat>, Children...>
    {
        typedef typename Matrify<MT, NT, Mat>::template run<T, Pack<MT, NT, Mat>, Children...> Sib;

        using Sib::rscat;
        using Sib::cscat;
        using Sib::rs;
        using Sib::cs;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            constexpr idx_type MR = MT<T>::def;
            constexpr idx_type NR = NT<T>::def;

            idx_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
            idx_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));
            m = round_up(m, MR);
            n = round_up(n, NR);

            MemoryPool& PackBuf = (Mat == MAT_A ? detail::BuffersForA : detail::BuffersForB);

            auto& pack_buffer = this->child.pack_buffer;
            T*& pack_ptr = this->child.pack_ptr;

            if (pack_ptr == NULL)
            {
                if (comm.master())
                {
                    idx_type scatter_size = size_as_type<stride_type,T>(2*m + 2*n);
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

template <template <typename> class MR, template <typename> class KR>
using MatrifyAndPackA = MatrifyAndPack<MR,KR,matrix_constants::MAT_A>;

template <template <typename> class KR, template <typename> class NR>
using MatrifyAndPackB = MatrifyAndPack<KR,NR,matrix_constants::MAT_B>;

template <typename T, typename Config>
using TensorGEMM =
    typename GEMM<Config,
                  PartitionNC<Config>,
                  PartitionKC<Config>,
                  MatrifyAndPackB<Config::template KR, Config::template NR>,
                  PartitionMC<Config>,
                  MatrifyAndPackA<Config::template MR, Config::template KR>,
                  MatrifyC<Config::template MR, Config::template NR>,
                  PartitionNR<Config>,
                  PartitionMR<Config>,
                  MicroKernel<Config>>::template run<T>;

}

#endif
