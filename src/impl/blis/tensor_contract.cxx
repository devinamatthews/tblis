#include "tblis.hpp"

using namespace std;
using namespace tblis::detail;

namespace tblis
{
namespace impl
{

namespace detail
{
    MemoryPool BuffersForScatter(4096);
}

template <typename T, idx_type MR, idx_type NR>
void BlockScatter(ThreadCommunicator& comm, tensor_matrix<T>& A, stride_type* rs, stride_type* cs, stride_type* rscat, stride_type* cscat)
{
    idx_type m = A.length(0);
    idx_type n = A.length(1);

    idx_type first, last;
    std::tie(first, last, std::ignore) = comm.distribute_over_threads(m, MR);

    A.length(0, last-first);
    A.shift_down(0, first);
    A.template fill_block_scatter<MR>(0, rs+first/MR, rscat+first);
    A.shift_up(0, first);
    A.length(0, m);

    std::tie(first, last, std::ignore) = comm.distribute_over_threads(n, NR);

    A.length(1, last-first);
    A.shift_down(1, first);
    A.template fill_block_scatter<NR>(1, cs+first/NR, cscat+first);
    A.shift_up(1, first);
    A.length(1, n);

    comm.barrier();
}

template <idx_type MR, idx_type NR, int Mat> struct MatrifyAndRun;

template <idx_type MR, idx_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_A>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        //BlockScatter<T,MR,NR>(comm, A, parent.rs, parent.cs, parent.rscat, parent.cscat);
        A.template fill_block_scatter<MR>(0, parent.rs, parent.rscat);
        A.template fill_block_scatter<NR>(1, parent.cs, parent.cscat);
        block_scatter_matrix<T,MR,NR> M(A.length(0), A.length(1), A.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(A.length(0), A.length(1), A.data(), parent.rscat, parent.cscat);
        parent.child(comm, alpha, M, B, beta, C);
    }
};

template <idx_type MR, idx_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_B>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        //BlockScatter<T,MR,NR>(comm, B, parent.rs, parent.cs, parent.rscat, parent.cscat);
        B.template fill_block_scatter<MR>(0, parent.rs, parent.rscat);
        B.template fill_block_scatter<NR>(1, parent.cs, parent.cscat);
        block_scatter_matrix<T,MR,NR> M(B.length(0), B.length(1), B.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(B.length(0), B.length(1), B.data(), parent.rscat, parent.cscat);
        parent.child(comm, alpha, A, M, beta, C);
    }
};

template <idx_type MR, idx_type NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_C>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        //BlockScatter<T,MR,NR>(comm, C, parent.rs, parent.cs, parent.rscat, parent.cscat);
        C.template fill_block_scatter<MR>(0, parent.rs, parent.rscat);
        C.template fill_block_scatter<NR>(1, parent.cs, parent.cscat);
        block_scatter_matrix<T,MR,NR> M(C.length(0), C.length(1), C.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(C.length(0), C.length(1), C.data(), parent.rscat, parent.cscat);
        parent.child(comm, alpha, A, B, beta, M);
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
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
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

            MatrifyAndRun<MR,NR,Mat>(*this, comm, alpha, A, B, beta, C);
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
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            constexpr idx_type MR = MT<T>::def;
            constexpr idx_type NR = NT<T>::def;

            idx_type m = (Mat == MAT_A ? A.length(0) : Mat == MAT_B ? B.length(0) : C.length(0));
            idx_type n = (Mat == MAT_A ? A.length(1) : Mat == MAT_B ? B.length(1) : C.length(1));
            m = round_up(m, MR);
            n = round_up(n, NR);

            MemoryPool& PackBuf = (Mat == MAT_A ? BuffersForA : BuffersForB);

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

            Sib::operator()(comm, alpha, A, B, beta, C);
        }
    };
};

template <template <typename> class MR, template <typename> class KR>
using MatrifyAndPackA = MatrifyAndPack<MR,KR,matrix_constants::MAT_A>;

template <template <typename> class KR, template <typename> class NR>
using MatrifyAndPackB = MatrifyAndPack<KR,NR,matrix_constants::MAT_B>;

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
int tensor_contract_blis_int(const std::vector<idx_type>& len_M,
                             const std::vector<idx_type>& len_N,
                             const std::vector<idx_type>& len_K,
                             T alpha, const T* A, const std::vector<stride_type>& stride_M_A,
                                                  const std::vector<stride_type>& stride_K_A,
                                      const T* B, const std::vector<stride_type>& stride_K_B,
                                                  const std::vector<stride_type>& stride_N_B,
                             T  beta,       T* C, const std::vector<stride_type>& stride_M_C,
                                                  const std::vector<stride_type>& stride_N_C)
{
    tensor_matrix<T> at(len_M, len_K, const_cast<T*>(A), stride_M_A, stride_K_A);
    tensor_matrix<T> bt(len_K, len_N, const_cast<T*>(B), stride_K_B, stride_N_B);
    tensor_matrix<T> ct(len_M, len_N,                C , stride_M_C, stride_N_C);

    typename GEMM<Config, 4, 0, 1, 9, 8, -1,
                  PartitionN<Config::template NC>,
                  PartitionK<Config::template KC>,
                  MatrifyAndPackB<Config::template KR, Config::template NR>,
                  PartitionM<Config::template MC>,
                  MatrifyAndPackA<Config::template MR, Config::template KR>,
                  MatrifyC<Config::template MR, Config::template NR>,
                  PartitionN<Config::template NR>,
                  PartitionM<Config::template MR>,
                  MicroKernel<Config>>::template run<T> gemm;

    gemm(alpha, at, bt, beta, ct);

    return 0;
}

template <typename T>
int tensor_contract_blis(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                  const const_tensor_view<T>& B, const std::string& idx_B,
                         T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    unsigned dim_A = A.dimension();
    unsigned dim_B = B.dimension();
    unsigned dim_C = C.dimension();

    unsigned dim_M = (dim_A+dim_C-dim_B)/2;
    unsigned dim_N = (dim_B+dim_C-dim_A)/2;
    unsigned dim_K = (dim_A+dim_B-dim_C)/2;

    vector<idx_type> len_M(dim_M);
    vector<idx_type> len_N(dim_N);
    vector<idx_type> len_K(dim_K);

    vector<stride_type> stride_M_A(dim_M);
    vector<stride_type> stride_M_C(dim_M);
    vector<stride_type> stride_N_B(dim_N);
    vector<stride_type> stride_N_C(dim_N);
    vector<stride_type> stride_K_A(dim_K);
    vector<stride_type> stride_K_B(dim_K);

    unsigned i_M = 0;
    unsigned i_N = 0;
    unsigned i_K = 0;

    for (unsigned i = 0;i < dim_A;i++)
    {
        for (unsigned j = 0;j < dim_B;j++)
        {
            if (idx_A[i] == idx_B[j])
            {
                idx_type len = A.length(i);
                if (len > 1)
                {
                    stride_K_A[i_K] = A.stride(i);
                    stride_K_B[i_K] = B.stride(j);
                    len_K[i_K] = len;
                    i_K++;
                }
            }
        }
    }
    dim_K = i_K;
    stride_K_A.resize(dim_K);
    stride_K_B.resize(dim_K);
    len_K.resize(dim_K);

    for (unsigned i = 0;i < dim_A;i++)
    {
        for (unsigned j = 0;j < dim_C;j++)
        {
            if (idx_A[i] == idx_C[j])
            {
                idx_type len = A.length(i);
                if (len > 1)
                {
                    stride_M_A[i_M] = A.stride(i);
                    stride_M_C[i_M] = C.stride(j);
                    len_M[i_M] = len;
                    i_M++;
                }
            }
        }
    }
    dim_M = i_M;
    stride_M_A.resize(dim_M);
    stride_M_C.resize(dim_M);
    len_M.resize(dim_M);

    for (unsigned i = 0;i < dim_B;i++)
    {
        for (unsigned j = 0;j < dim_C;j++)
        {
            if (idx_B[i] == idx_C[j])
            {
                idx_type len = B.length(i);
                if (len > 1)
                {
                    stride_N_B[i_N] = B.stride(i);
                    stride_N_C[i_N] = C.stride(j);
                    len_N[i_N] = len;
                    i_N++;
                }
            }
        }
    }
    dim_N = i_N;
    stride_N_B.resize(dim_N);
    stride_N_C.resize(dim_N);
    len_N.resize(dim_N);

    string idx_M, idx_N, idx_K;
    for (unsigned i = 0;i < dim_M;i++) idx_M.push_back(i);
    for (unsigned i = 0;i < dim_N;i++) idx_N.push_back(i);
    for (unsigned i = 0;i < dim_K;i++) idx_K.push_back(i);

    sort(idx_M.begin(), idx_M.end(),
    [&](char a, char b)
    {
        return stride_M_C[a] == stride_M_C[b] ?
               stride_M_A[a]  < stride_M_A[b] :
               stride_M_C[a]  < stride_M_C[b];
    });

    sort(idx_N.begin(), idx_N.end(),
    [&](char a, char b)
    {
        return stride_N_C[a] == stride_N_C[b] ?
               stride_N_B[a]  < stride_N_B[b] :
               stride_N_C[a]  < stride_N_C[b];
    });

    sort(idx_K.begin(), idx_K.end(),
    [&](char a, char b)
    {
        return stride_K_A[a] == stride_K_A[b] ?
               stride_K_B[a]  < stride_K_B[b] :
               stride_K_A[a]  < stride_K_A[b];
    });

    vector<stride_type> stride_M_Ar(dim_M);
    vector<stride_type> stride_M_Cr(dim_M);
    vector<stride_type> stride_N_Br(dim_N);
    vector<stride_type> stride_N_Cr(dim_N);
    vector<stride_type> stride_K_Ar(dim_K);
    vector<stride_type> stride_K_Br(dim_K);
    vector<idx_type> len_Mr(dim_M);
    vector<idx_type> len_Nr(dim_N);
    vector<idx_type> len_Kr(dim_K);

    for (unsigned i = 0;i < dim_M;i++)
    {
        stride_M_Ar[i] = stride_M_A[idx_M[i]];
        stride_M_Cr[i] = stride_M_C[idx_M[i]];
        len_Mr[i] = len_M[idx_M[i]];
    }

    for (unsigned i = 0;i < dim_N;i++)
    {
        stride_N_Br[i] = stride_N_B[idx_N[i]];
        stride_N_Cr[i] = stride_N_C[idx_N[i]];
        len_Nr[i] = len_N[idx_N[i]];
    }

    for (unsigned i = 0;i < dim_K;i++)
    {
        stride_K_Ar[i] = stride_K_A[idx_K[i]];
        stride_K_Br[i] = stride_K_B[idx_K[i]];
        len_Kr[i] = len_K[idx_K[i]];
    }

    if (dim_N > 0 && stride_N_Cr[0] == 1)
    {
        tensor_contract_blis_int(len_Nr, len_Mr, len_Kr,
                                 alpha, B.data(), stride_N_Br, stride_K_Br,
                                        A.data(), stride_K_Ar, stride_M_Ar,
                                  beta, C.data(), stride_N_Cr, stride_M_Cr);
    }
    else
    {
        tensor_contract_blis_int(len_Mr, len_Nr, len_Kr,
                                 alpha, A.data(), stride_M_Ar, stride_K_Ar,
                                        B.data(), stride_K_Br, stride_N_Br,
                                  beta, C.data(), stride_M_Cr, stride_N_Cr);
    }

    return 0;
}


#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_contract_blis<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                     const const_tensor_view<T>& B, const std::string& idx_B, \
                            T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
