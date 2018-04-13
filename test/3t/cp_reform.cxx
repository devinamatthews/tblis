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

    len_vector len_A;
    random_lengths(N/sizeof(T), ndim_A+1, len_A);
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
void cp_reform_ref(T alpha, const vector<matrix<T>>& U,
                            const vector<const label_type*>& idx_U,
                   T  beta, varray<T>& A, const label_vector& idx_A)
{
    unsigned ndim_A = A.dimension();

    len_type len_r;
    len_vector len_m(ndim_A);
    stride_vector stride_U_r(ndim_A);
    stride_vector stride_A_m(ndim_A);
    stride_vector stride_U_m(ndim_A);

    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (idx_U[i][0] == idx_A[j])
            {
                len_r = U[i].length(1);
                len_m[i] = A.length(j);
                stride_A_m[i] = A.stride(j);
                stride_U_m[i] = U[i].stride(0);
                stride_U_r[i] = U[i].stride(1);
                break;
            }
            else if (idx_U[i][1] == idx_A[j])
            {
                len_r = U[i].length(0);
                len_m[i] = A.length(j);
                stride_A_m[i] = A.stride(j);
                stride_U_m[i] = U[i].stride(1);
                stride_U_r[i] = U[i].stride(0);
                break;
            }
        }
    }

    auto ptr_A = A.data();
    viterator<1> iter_m(len_m, stride_A_m);
    while (iter_m.next(ptr_A))
    {
        T tmp0 = T(0);

        if (alpha != T(0))
        {
            for (len_type r = 0;r < len_r;r++)
            {
                T tmp1 = alpha;
                for (unsigned i = 0;i < ndim_A;i++)
                    tmp1 *= U[i].data()[iter_m.position()[i]*stride_U_m[i] + r*stride_U_r[i]];
                tmp0 += tmp1;
            }
        }

        if (beta == T(0))
        {
            (*ptr_A) = tmp0;
        }
        else
        {
            (*ptr_A) = tmp0 + beta*(*ptr_A);
        }
    }
}

REPLICATED_TEMPLATED_TEST_CASE(cp_reform, R, T, all_types)
{
    vector<matrix<T>> U;
    varray<T> A, B, C;
    vector<label_vector> idx_U_;
    vector<const label_type*> idx_U;
    label_vector idx_A;

    T scale(10.0*random_unit<T>());

    random_cp(N, U, idx_U_, A, idx_A);

    for (auto& idx : idx_U_) idx_U.push_back(idx.data());

    U.resize(8);
    idx_U.resize(8, "");
    TENSOR_INFO(A);
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

    B.reset(A);
    cp_reform_ref<T>(scale, U, idx_U, scale, B, idx_A.data());

    C.reset(A);
    cp_reform<T>(scale, U, idx_U, scale, C, idx_A.data());

    add<T>(T(-1), B, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;

    check("REF", error, scale*neps);
}
