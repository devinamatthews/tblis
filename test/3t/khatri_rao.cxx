#include "../test.hpp"

/*
 * Creates a random tensor multiplication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_khatri_rao(stride_type N, vector<matrix<T>>& U,
                                      vector<label_vector>& idx_U,
                                      varray<T>& A, label_vector& idx_A)
{
    unsigned ndim_A = random_number(2,8);

    random_tensor(N, ndim_A, A);

    idx_A.resize(ndim_A);
    for (unsigned i = 0;i < ndim_A-1;i++)
        idx_A[i] = 'a'+i;
    idx_A[ndim_A-1] = 'r';

    random_shuffle(idx_A.begin(), idx_A.end());

    unsigned r = 0;
    for (;r < ndim_A;r++)
        if (idx_A[r] == 'r') break;

    U.resize(ndim_A-1);
    idx_U.resize(ndim_A-1);
    for (unsigned i = 0;i < ndim_A-1;i++)
    {
        idx_U[i] = {label_type('a'+i), label_type('r')};
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (idx_A[j] != 'a'+i) continue;
            random_matrix(N, A.length(j), A.length(r), U[i]);
        }
    }
}

template <typename T>
void khatri_rao_ref(T alpha, const vector<matrix<T>>& U,
                             const vector<const label_type*>& idx_U,
                    T  beta, varray<T>& A, const label_vector& idx_A)
{
    unsigned ndim_A = A.dimension();
    unsigned ndim_m = U.size();

    len_type len_r;
    stride_type stride_A_r;
    stride_vector stride_U_r(ndim_m);

    unsigned idx_A_r;
    for (unsigned j = 0;j < ndim_A;j++)
    {
        unsigned nmatch = 0;
        for (unsigned i = 0;i < ndim_m;i++)
        {
            if (idx_U[i][0] == idx_A[j] ||
                idx_U[i][1] == idx_A[j])
                nmatch++;
        }

        if (nmatch == ndim_m)
        {
            idx_A_r = j;
            len_r = A.length(j);
            stride_A_r = A.stride(j);
            break;
        }
    }

    len_vector len_m(ndim_m);
    stride_vector stride_A_m(ndim_m);
    stride_vector stride_U_m(ndim_m);

    for (unsigned i = 0;i < ndim_m;i++)
    {
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (j == idx_A_r) continue;

            if (idx_U[i][0] == idx_A[j])
            {
                len_m[i] = A.length(j);
                stride_A_m[i] = A.stride(j);
                stride_U_m[i] = U[i].stride(0);
                stride_U_r[i] = U[i].stride(1);
                break;
            }
            else if (idx_U[i][1] == idx_A[j])
            {
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
        for (len_type r = 0;r < len_r;r++)
        {
            T tmp = alpha;
            for (unsigned i = 0;i < ndim_m;i++)
                tmp *= U[i].data()[iter_m.position()[i]*stride_U_m[i] + r*stride_U_r[i]];

            if (alpha == T(0))
            {
                if (beta == T(0))
                {
                    ptr_A[r*stride_A_r] = T(0);
                }
                else
                {
                    ptr_A[r*stride_A_r] *= beta;
                }
            }
            else
            {
                if (beta == T(0))
                {
                    ptr_A[r*stride_A_r] = tmp;
                }
                else
                {
                    ptr_A[r*stride_A_r] = tmp + beta*ptr_A[r*stride_A_r];
                }
            }
        }
    }
}

REPLICATED_TEMPLATED_TEST_CASE(khatri_rao, R, T, all_types)
{
    vector<matrix<T>> U;
    varray<T> A, B, C;
    vector<label_vector> idx_U_;
    vector<const label_type*> idx_U;
    label_vector idx_A;

    T scale(10.0*random_unit<T>());

    random_khatri_rao(N, U, idx_U_, A, idx_A);

    for (auto& idx : idx_U_) idx_U.push_back(idx.data());

    U.resize(7);
    idx_U.resize(7, "");
    TENSOR_INFO(A);
    TENSOR_INFO(U[0]);
    TENSOR_INFO(U[1]);
    TENSOR_INFO(U[2]);
    TENSOR_INFO(U[3]);
    TENSOR_INFO(U[4]);
    TENSOR_INFO(U[5]);
    TENSOR_INFO(U[6]);
    U.resize(idx_U_.size());
    idx_U.resize(idx_U_.size());

    auto neps = prod(A.lengths())*A.dimension();

    B.reset(A);
    khatri_rao_ref<T>(scale, U, idx_U, scale, B, idx_A.data());

    C.reset(A);
    khatri_rao<T>(scale, U, idx_U, scale, C, idx_A.data());

    add<T>(T(-1), B, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;

    check("REF", error, scale*neps);
}
