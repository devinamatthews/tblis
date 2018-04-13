#include "../test.hpp"

/*
 * Creates a random tensor multiplication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_cp(stride_type N, vector<matrix<T>>& U,
                              vector<label_vector>& idx_U,
                              varray<T>& A, label_vector& idx_A)
{
    unsigned ndim_A = random_number(2,8);
    ndim_A = 5;

    len_vector len_A;
    random_lengths(N/sizeof(T), ndim_A+1, len_A);
    len_A = {10, 10, 10, 10, 10, 1};
    len_type len_r = len_A.back();
    len_A.pop_back();

    random_tensor(N, ndim_A, len_A, A);

    idx_A.resize(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
        idx_A[i] = 'a'+i;

    random_shuffle(idx_A.begin(), idx_A.end());

    U.resize(ndim_A);
    idx_U.resize(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        idx_U[i] = {label_type('a'+i), label_type('r')};
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (idx_A[j] != 'a'+i) continue;
            random_matrix(N, A.length(j), len_r, U[i]);
        }
    }
}

template <typename T>
void cp_gradient_ref(const varray<T>& A, const label_vector& idx_A,
                     const vector<matrix<T>>& U_,
                     const vector<const label_type*>& idx_U_,
                     matrix<T>& G, const label_vector& idx_G)
{
    auto len_K = A.lengths();
    auto idx_K = idx_A;
    label_type idx_grad;

    for (unsigned i = 0;i < A.dimension();i++)
    {
        if (idx_G[0] == idx_A[i])
        {
            len_K[i] = G.length(1);
            idx_K[i] = 'r';
            idx_grad = idx_A[i];
            break;
        }
        else if (idx_G[1] == idx_A[i])
        {
            len_K[i] = G.length(0);
            idx_K[i] = 'r';
            idx_grad = idx_A[i];
            break;
        }
    }

    vector<matrix_view<const T>> U;
    vector<const label_type*> idx_U;

    for (unsigned i = 0;i < A.dimension();i++)
    {
        if (idx_U_[i][0] == idx_grad ||
            idx_U_[i][1] == idx_grad) continue;

        U.emplace_back(U_[i]);
        idx_U.push_back(idx_U_[i]);
    }

    varray<T> K(len_K);
    khatri_rao<T>(T(1), U, idx_U, T(0), K, idx_K.data());
    mult<T>(T(1), A, idx_A.data(), K, idx_K.data(), T(0), vary(G), idx_G.data());
}

REPLICATED_TEMPLATED_TEST_CASE(cp_gradient, R, T, all_types)
{
    vector<matrix<T>> U;
    varray<T> A;
    matrix<T> B, C;
    vector<label_vector> idx_U_;
    vector<const label_type*> idx_U;
    label_vector idx_A;

    random_cp(N, U, idx_U_, A, idx_A);

    //A = 1;
    //U[0] = 1;
    //U[1] = 1;
    //U[2] = 1;

    unsigned dim = random_number(U.size()-1);
    auto G = U[dim];
    auto idx_G = idx_U_[dim];

    for (auto& idx : idx_U_) idx_U.push_back(idx.data());

    U.resize(8);
    idx_U.resize(8, "");
    TENSOR_INFO(A);
    TENSOR_INFO(G);
    TENSOR_INFO(U[0]);
    TENSOR_INFO(U[1]);
    TENSOR_INFO(U[2]);
    TENSOR_INFO(U[3]);
    TENSOR_INFO(U[4]);
    TENSOR_INFO(U[5]);
    TENSOR_INFO(U[6]);
    TENSOR_INFO(U[7]);
    U.resize(idx_U_.size());
    idx_U.resize(idx_U_.size());

    auto neps = prod(A.lengths())*U[0].length(1)*A.dimension();

    B.reset(G.lengths(), uninitialized);
    cp_gradient_ref<T>(A, idx_A, U, idx_U, B, idx_G);

    internal::cp_impl = NAIVE;
    C.reset(G.lengths(), uninitialized);
    cp_gradient<T>(A, idx_A.data(), U, idx_U, C, idx_G.data());

    //PRINT_TENSOR(A);
    //PRINT_MATRIX(U[0]);
    //PRINT_MATRIX(U[1]);
    //PRINT_MATRIX(U[2]);
    //PRINT_MATRIX(B);
    //PRINT_MATRIX(C);

    add<T>(T(-1), B, T(1), C);
    T error = reduce<T>(REDUCE_NORM_2, C).first;

    check("NAIVE", error, neps);

    internal::cp_impl = DIRECT;
    C.reset(G.lengths(), uninitialized);
    cp_gradient<T>(A, idx_A.data(), U, idx_U, C, idx_G.data());

    //PRINT_TENSOR(A);
    //PRINT_MATRIX(U[0]);
    //PRINT_MATRIX(U[1]);
    //PRINT_MATRIX(U[2]);
    //PRINT_MATRIX(B);
    //PRINT_MATRIX(C);

    add<T>(T(-1), B, T(1), C);
    error = reduce<T>(REDUCE_NORM_2, C).first;

    check("DIRECT", error, neps);

    internal::cp_impl = PHAN;
    C.reset(G.lengths(), uninitialized);
    cp_gradient<T>(A, idx_A.data(), U, idx_U, C, idx_G.data());

    add<T>(T(-1), B, T(1), C);
    error = reduce<T>(REDUCE_NORM_2, C).first;

    check("PHAN", error, neps);
}
