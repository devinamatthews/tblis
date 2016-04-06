#include "tblis.hpp"
#include "impl/tensor_impl.hpp"
#include "util/util.hpp"

#define SCATTER_MATRIX 0
#define TENSOR_SCATTER 1
#define TENSOR_BLOCK_SCATTER 2
#define TENSOR 3

#define IMPL_TYPE TENSOR

using namespace std;
using namespace tblis::util;
using namespace tblis::blis_like;

namespace tblis
{
namespace impl
{

namespace detail
{
    MemoryPool BuffersForScatter(BLIS_HEAP_ADDR_ALIGN_SIZE);
}

#if IMPL_TYPE == SCATTER_MATRIX

template <typename T>
int tensor_contract_blis_int(const std::vector<dim_t>& len_M,
                             const std::vector<dim_t>& len_N,
                             const std::vector<dim_t>& len_K,
                             T alpha, const T* A, const std::vector<inc_t>& stride_M_A,
                                                  const std::vector<inc_t>& stride_K_A,
                                      const T* B, const std::vector<inc_t>& stride_K_B,
                                                  const std::vector<inc_t>& stride_N_B,
                             T  beta,       T* C, const std::vector<inc_t>& stride_M_C,
                                                  const std::vector<inc_t>& stride_N_C)
{
    dim_t m = prod(len_M);
    dim_t n = prod(len_N);
    dim_t k = prod(len_K);

    vector<inc_t> scat_M_A(m);
    vector<inc_t> scat_M_C(m);
    vector<inc_t> scat_N_B(n);
    vector<inc_t> scat_N_C(n);
    vector<inc_t> scat_K_A(k);
    vector<inc_t> scat_K_B(k);

    Iterator<> it_M_A(len_M, stride_M_A);
    inc_t idx_M_A = 0;
    for (dim_t i = 0;it_M_A.next(idx_M_A);i++) scat_M_A[i] = idx_M_A;

    Iterator<> it_M_C(len_M, stride_M_C);
    inc_t idx_M_C = 0;
    for (dim_t i = 0;it_M_C.next(idx_M_C);i++) scat_M_C[i] = idx_M_C;

    Iterator<> it_N_B(len_N, stride_N_B);
    inc_t idx_N_B = 0;
    for (dim_t i = 0;it_N_B.next(idx_N_B);i++) scat_N_B[i] = idx_N_B;

    Iterator<> it_N_C(len_N, stride_N_C);
    inc_t idx_N_C = 0;
    for (dim_t i = 0;it_N_C.next(idx_N_C);i++) scat_N_C[i] = idx_N_C;

    Iterator<> it_K_A(len_K, stride_K_A);
    inc_t idx_K_A = 0;
    for (dim_t i = 0;it_K_A.next(idx_K_A);i++) scat_K_A[i] = idx_K_A;

    Iterator<> it_K_B(len_K, stride_K_B);
    inc_t idx_K_B = 0;
    for (dim_t i = 0;it_K_B.next(idx_K_B);i++) scat_K_B[i] = idx_K_B;

    ScatterMatrix<T> as(m, k, const_cast<T*>(A), scat_M_A.data(), scat_K_A.data());
    ScatterMatrix<T> bs(k, n, const_cast<T*>(B), scat_K_B.data(), scat_N_B.data());
    ScatterMatrix<T> cs(m, n,                C , scat_M_C.data(), scat_N_C.data());

    tblis_gemm_int(alpha, as, bs, beta, cs);

    return 0;
}

#elif IMPL_TYPE == TENSOR_SCATTER

template <typename T>
int tensor_contract_blis_int(const std::vector<dim_t>& len_M,
                             const std::vector<dim_t>& len_N,
                             const std::vector<dim_t>& len_K,
                             T alpha, const T* A, const std::vector<inc_t>& stride_M_A,
                                                  const std::vector<inc_t>& stride_K_A,
                                      const T* B, const std::vector<inc_t>& stride_K_B,
                                                  const std::vector<inc_t>& stride_N_B,
                             T  beta,       T* C, const std::vector<inc_t>& stride_M_C,
                                                  const std::vector<inc_t>& stride_N_C)
{
    dim_t m = prod(len_M);
    dim_t n = prod(len_N);
    dim_t k = prod(len_K);

    vector<inc_t> scat_M_A(m);
    vector<inc_t> scat_M_C(m);
    vector<inc_t> scat_N_B(n);
    vector<inc_t> scat_N_C(n);
    vector<inc_t> scat_K_A(k);
    vector<inc_t> scat_K_B(k);

    TensorMatrix<T> at(len_M, len_K, const_cast<T*>(A), stride_M_A, stride_K_A);
    TensorMatrix<T> bt(len_K, len_N, const_cast<T*>(B), stride_K_B, stride_N_B);
    TensorMatrix<T> ct(len_M, len_N,                C , stride_M_C, stride_N_C);

    at.row_scatter(scat_M_A.data());
    at.col_scatter(scat_K_A.data());
    bt.row_scatter(scat_K_B.data());
    bt.col_scatter(scat_N_B.data());
    ct.row_scatter(scat_M_C.data());
    ct.col_scatter(scat_N_C.data());

    //printf("M_A: "); for (int i = 0;i < m;i++) printf("%6ld ", scat_M_A[i]); printf("\n");
    //printf("M_C: "); for (int i = 0;i < m;i++) printf("%6ld ", scat_M_C[i]); printf("\n");
    //printf("N_B: "); for (int i = 0;i < n;i++) printf("%6ld ", scat_N_B[i]); printf("\n");
    //printf("N_C: "); for (int i = 0;i < n;i++) printf("%6ld ", scat_N_C[i]); printf("\n");
    //printf("K_A: "); for (int i = 0;i < k;i++) printf("%6ld ", scat_K_A[i]); printf("\n");
    //printf("K_B: "); for (int i = 0;i < k;i++) printf("%6ld ", scat_K_B[i]); printf("\n");

    ScatterMatrix<T> as(m, k, const_cast<T*>(A), scat_M_A.data(), scat_K_A.data());
    ScatterMatrix<T> bs(k, n, const_cast<T*>(B), scat_K_B.data(), scat_N_B.data());
    ScatterMatrix<T> cs(m, n,                C , scat_M_C.data(), scat_N_C.data());

    tblis_gemm_int(alpha, as, bs, beta, cs);

    return 0;
}

#elif IMPL_TYPE == TENSOR_BLOCK_SCATTER

template <typename T>
int tensor_contract_blis_int(const std::vector<dim_t>& len_M,
                             const std::vector<dim_t>& len_N,
                             const std::vector<dim_t>& len_K,
                             T alpha, const T* A, const std::vector<inc_t>& stride_M_A,
                                                  const std::vector<inc_t>& stride_K_A,
                                      const T* B, const std::vector<inc_t>& stride_K_B,
                                                  const std::vector<inc_t>& stride_N_B,
                             T  beta,       T* C, const std::vector<inc_t>& stride_M_C,
                                                  const std::vector<inc_t>& stride_N_C)
{
    constexpr dim_t M = MR<T>::def;
    constexpr dim_t N = NR<T>::def;

    dim_t m = prod(len_M);
    dim_t n = prod(len_N);
    dim_t k = prod(len_K);

    vector<inc_t> scat_M_A(m);
    vector<inc_t> scat_M_C(m);
    vector<inc_t> scat_N_B(n);
    vector<inc_t> scat_N_C(n);
    vector<inc_t> scat_K_A(k);
    vector<inc_t> scat_K_B(k);

    vector<inc_t> s_M_A(m);
    vector<inc_t> s_M_C(m);
    vector<inc_t> s_N_B(n);
    vector<inc_t> s_N_C(n);

    TensorMatrix<T> at(len_M, len_K, const_cast<T*>(A), stride_M_A, stride_K_A);
    TensorMatrix<T> bt(len_K, len_N, const_cast<T*>(B), stride_K_B, stride_N_B);
    TensorMatrix<T> ct(len_M, len_N,                C , stride_M_C, stride_N_C);

    at.template row_block_scatter<M>(s_M_A.data(), scat_M_A.data());
    bt.template col_block_scatter<N>(s_N_B.data(), scat_N_B.data());
    ct.template row_block_scatter<M>(s_M_C.data(), scat_M_C.data());
    ct.template col_block_scatter<N>(s_N_C.data(), scat_N_C.data());
    at.col_scatter(scat_K_A.data());
    bt.row_scatter(scat_K_B.data());

    /*
    printf("\n");
    for (int i = 0;i < m;i++) printf("%6ld ", scat_M_A[i]); printf("\n");
    for (int i = 0;i < m-1;i++) printf("%6ld ", scat_M_A[i+1]-scat_M_A[i]); printf("     0\n");
    for (int i = 0;i < (m+M-1)/M;i++) printf("%*ld ", 7*M-1, s_M_A[i]); printf("\n");
    printf("\n");
    for (int i = 0;i < n;i++) printf("%6ld ", scat_N_B[i]); printf("\n");
    for (int i = 0;i < n-1;i++) printf("%6ld ", scat_N_B[i+1]-scat_N_B[i]); printf("     0\n");
    for (int i = 0;i < (n+N-1)/N;i++) printf("%*ld ", 7*N-1, s_N_B[i]); printf("\n");
    printf("\n");
    for (int i = 0;i < m;i++) printf("%6ld ", scat_M_C[i]); printf("\n");
    for (int i = 0;i < m-1;i++) printf("%6ld ", scat_M_C[i+1]-scat_M_C[i]); printf("     0\n");
    for (int i = 0;i < (m+M-1)/M;i++) printf("%*ld ", 7*M-1, s_M_C[i]); printf("\n");
    printf("\n");
    for (int i = 0;i < n;i++) printf("%6ld ", scat_N_C[i]); printf("\n");
    for (int i = 0;i < n-1;i++) printf("%6ld ", scat_N_C[i+1]-scat_N_C[i]); printf("     0\n");
    for (int i = 0;i < (n+N-1)/N;i++) printf("%*ld ", 7*N-1, s_N_C[i]); printf("\n");
    printf("\n");
    for (int i = 0;i < k;i++) printf("%6ld ", scat_K_A[i]); printf("\n");
    for (int i = 0;i < k;i++) printf("%6ld ", scat_K_B[i]); printf("\n");
    printf("\n");
    */

    BlockScatterMatrix<T,M,0> abs(m, k, const_cast<T*>(A), s_M_A.data(),         NULL, scat_M_A.data(), scat_K_A.data());
    BlockScatterMatrix<T,0,N> bbs(k, n, const_cast<T*>(B),         NULL, s_N_B.data(), scat_K_B.data(), scat_N_B.data());
    BlockScatterMatrix<T,M,N> cbs(m, n,                C , s_M_C.data(), s_N_C.data(), scat_M_C.data(), scat_N_C.data());

    tblis_gemm_int(alpha, abs, bbs, beta, cbs);

    return 0;
}

#elif IMPL_TYPE == TENSOR

template <typename T, dim_t MR, dim_t NR>
void BlockScatter(ThreadCommunicator& comm, TensorMatrix<T>& A, inc_t* rs, inc_t* cs, inc_t* rscat, inc_t* cscat)
{
    dim_t m = A.length();
    dim_t n = A.width();

    dim_t first, last;
    std::tie(first, last) = comm.distribute_over_threads(m, MR);

    A.length(last-first);
    A.shift_down(first);
    A.template row_block_scatter<MR>(rs+first/MR, rscat+first);
    A.shift_up(first);
    A.length(m);

    std::tie(first, last) = comm.distribute_over_threads(n, NR);

    A.width(last-first);
    A.shift_right(first);
    A.template col_block_scatter<NR>(cs+first/NR, cscat+first);
    A.shift_left(first);
    A.width(n);

    comm.barrier();
}

template <dim_t MR, dim_t NR, int Mat> struct MatrifyAndRun;

template <dim_t MR, dim_t NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_A>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        BlockScatter<T,MR,NR>(comm, A, parent.rs, parent.cs, parent.rscat, parent.cscat);
        BlockScatterMatrix<T,MR,NR> M(A.length(), A.width(), A.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(A.length(), A.width(), A.data(), parent.rscat, parent.cscat);
        parent.child(comm, alpha, M, B, beta, C);
    }
};

template <dim_t MR, dim_t NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_B>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        BlockScatter<T,MR,NR>(comm, B, parent.rs, parent.cs, parent.rscat, parent.cscat);
        BlockScatterMatrix<T,MR,NR> M(B.length(), B.width(), B.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(B.length(), B.width(), B.data(), parent.rscat, parent.cscat);
        parent.child(comm, alpha, A, M, beta, C);
    }
};

template <dim_t MR, dim_t NR> struct MatrifyAndRun<MR, NR, matrix_constants::MAT_C>
{
    template <typename T, typename Parent, typename MatrixA, typename MatrixB, typename MatrixC>
    MatrifyAndRun(Parent& parent, ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        BlockScatter<T,MR,NR>(comm, C, parent.rs, parent.cs, parent.rscat, parent.cscat);
        BlockScatterMatrix<T,MR,NR> M(C.length(), C.width(), C.data(), parent.rs, parent.cs, parent.rscat, parent.cscat);
        //ScatterMatrix<T> M(C.length(), C.width(), C.data(), parent.rscat, parent.cscat);
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

        MemoryPool::Block<inc_t> scat_buffer;
        inc_t* rscat = NULL;
        inc_t* cscat = NULL;
        inc_t* rs = NULL;
        inc_t* cs = NULL;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;

            constexpr dim_t MR = MT<T>::def;
            constexpr dim_t NR = NT<T>::def;

            dim_t m = (Mat == MAT_A ? A.length() : Mat == MAT_B ? B.length() : C.length());
            dim_t n = (Mat == MAT_A ? A.width()  : Mat == MAT_B ? B.width()  : C.width());

            if (rscat == NULL)
            {
                if (comm.master())
                {
                    scat_buffer = detail::BuffersForScatter.allocate<inc_t>(2*m + 2*n);
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

            constexpr dim_t MR = MT<T>::def;
            constexpr dim_t NR = NT<T>::def;

            dim_t m = (Mat == MAT_A ? A.length() : Mat == MAT_B ? B.length() : C.length());
            dim_t n = (Mat == MAT_A ? A.width()  : Mat == MAT_B ? B.width()  : C.width());
            m = util::round_up(m, MR);
            n = util::round_up(n, NR);

            MemoryPool& PackBuf = (Mat == MAT_A ? blis_like::detail::BuffersForA
                                                : blis_like::detail::BuffersForB);

            auto& pack_buffer = this->child.pack_buffer;
            T* pack_ptr = this->child.pack_ptr;

            if (pack_ptr == NULL)
            {
                if (comm.master())
                {
                    dim_t scatter_size = util::size_as_type<inc_t,T>(2*m + 2*n);
                    pack_buffer = PackBuf.allocate<T>(m*(n+TBLIS_MAX_UNROLL) + scatter_size);
                    pack_ptr = pack_buffer;
                }

                comm.broadcast(pack_ptr);

                rscat = util::convert_and_align<T,inc_t>(pack_ptr + m*n);
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

template <typename T>
int tensor_contract_blis_int(const std::vector<dim_t>& len_M,
                             const std::vector<dim_t>& len_N,
                             const std::vector<dim_t>& len_K,
                             T alpha, const T* A, const std::vector<inc_t>& stride_M_A,
                                                  const std::vector<inc_t>& stride_K_A,
                                      const T* B, const std::vector<inc_t>& stride_K_B,
                                                  const std::vector<inc_t>& stride_N_B,
                             T  beta,       T* C, const std::vector<inc_t>& stride_M_C,
                                                  const std::vector<inc_t>& stride_N_C)
{
    TensorMatrix<T> at(len_M, len_K, const_cast<T*>(A), stride_M_A, stride_K_A);
    TensorMatrix<T> bt(len_K, len_N, const_cast<T*>(B), stride_K_B, stride_N_B);
    TensorMatrix<T> ct(len_M, len_N,                C , stride_M_C, stride_N_C);

    dim_t jc_way = bli_read_nway_from_env( "BLIS_JC_NT" );
    dim_t ic_way = bli_read_nway_from_env( "BLIS_IC_NT" );
    dim_t jr_way = bli_read_nway_from_env( "BLIS_JR_NT" );
    dim_t ir_way = bli_read_nway_from_env( "BLIS_IR_NT" );
    dim_t nthread = jc_way*ic_way*jr_way*ir_way;

    //printf_locked("%d %d %d %d\n", jc_way, ic_way, jr_way, ir_way);

    GEMM<PartitionN<NC>,
         PartitionK<KC>,
         MatrifyAndPackB<KR,NR>,
         PartitionM<MC>,
         MatrifyAndPackA<MR,KR>,
         MatrifyC<MR,NR>,
         MacroKernel<MR,NR>>::run<T> gemm;

    gemm.template step<0>().distribute = jc_way;
    gemm.template step<1>().distribute = 1; //kc_way
    gemm.template step<4>().distribute = ic_way;
    gemm.template step<8>().distribute = jr_way;
    gemm.template step<9>().distribute = ir_way;

    parallelize(nthread,
    [=](ThreadCommunicator& comm) mutable
    {
        gemm(comm, alpha, at, bt, beta, ct);
    });

    return 0;
}

#endif

template <typename T>
int tensor_contract_blis(T alpha, const Tensor<T>& A, const std::string& idx_A,
                                  const Tensor<T>& B, const std::string& idx_B,
                         T  beta,       Tensor<T>& C, const std::string& idx_C)
{
    gint_t dim_A = A.dimension();
    gint_t dim_B = B.dimension();
    gint_t dim_C = C.dimension();

    gint_t dim_M = (dim_A+dim_C-dim_B)/2;
    gint_t dim_N = (dim_B+dim_C-dim_A)/2;
    gint_t dim_K = (dim_A+dim_B-dim_C)/2;

    vector<dim_t> len_M(dim_M);
    vector<dim_t> len_N(dim_N);
    vector<dim_t> len_K(dim_K);

    vector<inc_t> stride_M_A(dim_M);
    vector<inc_t> stride_M_C(dim_M);
    vector<inc_t> stride_N_B(dim_N);
    vector<inc_t> stride_N_C(dim_N);
    vector<inc_t> stride_K_A(dim_K);
    vector<inc_t> stride_K_B(dim_K);

    gint_t i_M = 0;
    gint_t i_N = 0;
    gint_t i_K = 0;

    for (gint_t i = 0;i < dim_A;i++)
    {
        for (gint_t j = 0;j < dim_B;j++)
        {
            if (idx_A[i] == idx_B[j])
            {
                dim_t len = A.length(i);
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

    for (gint_t i = 0;i < dim_A;i++)
    {
        for (gint_t j = 0;j < dim_C;j++)
        {
            if (idx_A[i] == idx_C[j])
            {
                dim_t len = A.length(i);
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

    for (gint_t i = 0;i < dim_B;i++)
    {
        for (gint_t j = 0;j < dim_C;j++)
        {
            if (idx_B[i] == idx_C[j])
            {
                dim_t len = B.length(i);
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
    for (gint_t i = 0;i < dim_M;i++) idx_M.push_back(i);
    for (gint_t i = 0;i < dim_N;i++) idx_N.push_back(i);
    for (gint_t i = 0;i < dim_K;i++) idx_K.push_back(i);

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

    vector<inc_t> stride_M_Ar(dim_M);
    vector<inc_t> stride_M_Cr(dim_M);
    vector<inc_t> stride_N_Br(dim_N);
    vector<inc_t> stride_N_Cr(dim_N);
    vector<inc_t> stride_K_Ar(dim_K);
    vector<inc_t> stride_K_Br(dim_K);
    vector<dim_t> len_Mr(dim_M);
    vector<dim_t> len_Nr(dim_N);
    vector<dim_t> len_Kr(dim_K);

    for (gint_t i = 0;i < dim_M;i++)
    {
        stride_M_Ar[i] = stride_M_A[idx_M[i]];
        stride_M_Cr[i] = stride_M_C[idx_M[i]];
        len_Mr[i] = len_M[idx_M[i]];
    }

    for (gint_t i = 0;i < dim_N;i++)
    {
        stride_N_Br[i] = stride_N_B[idx_N[i]];
        stride_N_Cr[i] = stride_N_C[idx_N[i]];
        len_Nr[i] = len_N[idx_N[i]];
    }

    for (gint_t i = 0;i < dim_K;i++)
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

template
int tensor_contract_blis<   float>(   float alpha, const Tensor<   float>& A, const std::string& idx_A,
                                                   const Tensor<   float>& B, const std::string& idx_B,
                                      float  beta,       Tensor<   float>& C, const std::string& idx_C);

template
int tensor_contract_blis<  double>(  double alpha, const Tensor<  double>& A, const std::string& idx_A,
                                                   const Tensor<  double>& B, const std::string& idx_B,
                                     double  beta,       Tensor<  double>& C, const std::string& idx_C);

template
int tensor_contract_blis<sComplex>(sComplex alpha, const Tensor<sComplex>& A, const std::string& idx_A,
                                                   const Tensor<sComplex>& B, const std::string& idx_B,
                                   sComplex  beta,       Tensor<sComplex>& C, const std::string& idx_C);

template
int tensor_contract_blis<dComplex>(dComplex alpha, const Tensor<dComplex>& A, const std::string& idx_A,
                                                   const Tensor<dComplex>& B, const std::string& idx_B,
                                   dComplex  beta,       Tensor<dComplex>& C, const std::string& idx_C);

}
}
