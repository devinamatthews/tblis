#include "tblis.hpp"

using namespace std;
using namespace stl_ext;
using namespace tblis::detail;

namespace tblis
{
namespace impl
{

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

    TensorGEMM<T, Config>()(alpha, at, bt, beta, ct);

    return 0;
}

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
int tensor_contract_blis_int(thread_communicator& comm,
                             const std::vector<idx_type>& len_M,
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

    TensorGEMM<T, Config>()(comm, alpha, at, bt, beta, ct);

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

    string idx_M = intersection(idx_A, idx_C);
    string idx_N = intersection(idx_B, idx_C);
    string idx_K = intersection(idx_A, idx_B);

    vector<idx_type> len_M = select_from(C.lengths(), idx_C, idx_M);
    vector<idx_type> len_N = select_from(C.lengths(), idx_C, idx_N);
    vector<idx_type> len_K = select_from(A.lengths(), idx_A, idx_K);

    vector<stride_type> stride_M_A = select_from(A.strides(), idx_A, idx_M);
    vector<stride_type> stride_M_C = select_from(C.strides(), idx_C, idx_M);
    vector<stride_type> stride_N_B = select_from(B.strides(), idx_B, idx_N);
    vector<stride_type> stride_N_C = select_from(C.strides(), idx_C, idx_N);
    vector<stride_type> stride_K_A = select_from(A.strides(), idx_A, idx_K);
    vector<stride_type> stride_K_B = select_from(B.strides(), idx_B, idx_K);

    vector<unsigned> reorder_M = MArray::range(dim_M);
    sort(reorder_M,
    [&](unsigned a, unsigned b)
    {
        return stride_M_C[a] == stride_M_C[b] ?
               stride_M_A[a]  < stride_M_A[b] :
               stride_M_C[a]  < stride_M_C[b];
    });

    vector<unsigned> reorder_N = MArray::range(dim_N);
    sort(reorder_N,
    [&](unsigned a, unsigned b)
    {
        return stride_N_C[a] == stride_N_C[b] ?
               stride_N_B[a]  < stride_N_B[b] :
               stride_N_C[a]  < stride_N_C[b];
    });

    vector<unsigned> reorder_K = MArray::range(dim_K);
    sort(reorder_K,
    [&](unsigned a, unsigned b)
    {
        return stride_K_A[a] == stride_K_A[b] ?
               stride_K_B[a]  < stride_K_B[b] :
               stride_K_A[a]  < stride_K_A[b];
    });

    tensor_contract_blis_int(permuted(len_M, reorder_M), permuted(len_N, reorder_N), permuted(len_K, reorder_K),
                             alpha, A.data(), permuted(stride_M_A, reorder_M), permuted(stride_K_A, reorder_K),
                                    B.data(), permuted(stride_K_B, reorder_K), permuted(stride_N_B, reorder_N),
                              beta, C.data(), permuted(stride_M_C, reorder_M), permuted(stride_N_C, reorder_N));

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
