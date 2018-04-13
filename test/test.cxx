#define CATCH_CONFIG_RUNNER
#include "test.hpp"

template <> const string& type_name<float>()
{
    static string name = "float";
    return name;
}

template <> const string& type_name<double>()
{
    static string name = "double";
    return name;
}

template <> const string& type_name<scomplex>()
{
    static string name = "scomplex";
    return name;
}

template <> const string& type_name<dcomplex>()
{
    static string name = "dcomplex";
    return name;
}

stride_type N = 1024*1024;
int R = 50;

template <typename T>
void gemm_ref(T alpha, matrix_view<const T> A,
                       matrix_view<const T> B,
              T  beta,       matrix_view<T> C)
{
    const T* ptr_A = A.data();
    const T* ptr_B = B.data();
          T* ptr_C = C.data();

    len_type m_A = A.length(0);
    len_type m_C = C.length(0);
    len_type n_B = B.length(1);
    len_type n_C = C.length(1);
    len_type k_A = A.length(1);
    len_type k_B = B.length(0);

    stride_type rs_A = A.stride(0);
    stride_type cs_A = A.stride(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);
    stride_type rs_C = C.stride(0);
    stride_type cs_C = C.stride(1);

    TBLIS_ASSERT(m_A == m_C);
    TBLIS_ASSERT(n_B == n_C);
    TBLIS_ASSERT(k_A == k_B);

    len_type m = m_A;
    len_type n = n_B;
    len_type k = k_A;

    for (len_type i = 0;i < m;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T tmp = T();

            if (alpha != T(0))
            {
                for (len_type ik = 0;ik < k;ik++)
                {
                    tmp += ptr_A[i*rs_A + ik*cs_A]*ptr_B[ik*rs_B + j*cs_B];
                }
            }

            if (beta == T(0))
            {
                ptr_C[i*rs_C + j*cs_C] = alpha*tmp;
            }
            else
            {
                ptr_C[i*rs_C + j*cs_C] = alpha*tmp + beta*ptr_C[i*rs_C + j*cs_C];
            }
        }
    }
}

#define FOREACH_TYPE(T) \
template \
void gemm_ref(T alpha, matrix_view<const T> A, \
                       matrix_view<const T> B, \
              T  beta,       matrix_view<T> C);
#include "configs/foreach_type.h"

template <typename T>
void gemm_ref(T alpha, matrix_view<const T> A,
                          row_view<const T> D,
                       matrix_view<const T> B,
              T  beta,       matrix_view<T> C)
{
    const T* ptr_A = A.data();
    const T* ptr_D = D.data();
    const T* ptr_B = B.data();
          T* ptr_C = C.data();

    len_type m_A = A.length(0);
    len_type m_C = C.length(0);
    len_type n_B = B.length(1);
    len_type n_C = C.length(1);
    len_type k_A = A.length(1);
    len_type k_B = B.length(0);
    len_type k_D = D.length();

    stride_type rs_A = A.stride(0);
    stride_type cs_A = A.stride(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);
    stride_type rs_C = C.stride(0);
    stride_type cs_C = C.stride(1);
    stride_type inc_D = D.stride();

    TBLIS_ASSERT(m_A == m_C);
    TBLIS_ASSERT(n_B == n_C);
    TBLIS_ASSERT(k_A == k_B);
    TBLIS_ASSERT(k_A == k_D);

    len_type m = m_A;
    len_type n = n_B;
    len_type k = k_A;

    for (len_type i = 0;i < m;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T tmp = T();

            if (alpha != T(0))
            {
                for (len_type ik = 0;ik < k;ik++)
                {
                    tmp += ptr_A[i*rs_A + ik*cs_A]*ptr_B[ik*rs_B + j*cs_B]*ptr_D[ik*inc_D];
                }
            }

            if (beta == T(0))
            {
                ptr_C[i*rs_C + j*cs_C] = alpha*tmp;
            }
            else
            {
                ptr_C[i*rs_C + j*cs_C] = alpha*tmp + beta*ptr_C[i*rs_C + j*cs_C];
            }
        }
    }
}

#define FOREACH_TYPE(T) \
template \
void gemm_ref(T alpha, matrix_view<const T> A, \
                          row_view<const T> D, \
                       matrix_view<const T> B, \
              T  beta,       matrix_view<T> C);
#include "configs/foreach_type.h"

/*
 * Creates a matrix whose total storage size is between N/4
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/16 and N/4. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void random_matrix(stride_type N, len_type m_min, len_type n_min, matrix<T>& t)
{
    auto len = random_product_constrained_sequence<len_type>(2, N/sizeof(T), len_vector{m_min, n_min});

    len_type m = (m_min > 0 ? m_min : random_number<len_type>(1, len[0]));
    len_type n = (n_min > 0 ? n_min : random_number<len_type>(1, len[1]));

    if (random_choice())
    {
        t.reset({m, n}, COLUMN_MAJOR);
    }
    else
    {
        t.reset({m, n}, ROW_MAJOR);
    }

    T* data = t.data();
    miterator<2> it(t.lengths(), t.strides());
    while (it.next(data)) *data = random_unit<T>();
}

#define FOREACH_TYPE(T) \
template void random_matrix(stride_type N, len_type m_min, len_type n_min, matrix<T>& t);
#include "configs/foreach_type.h"

/*
 * Creates a matrix, whose total storage size is between N/4
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/16 and N/4. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void random_matrix(stride_type N, matrix<T>& t)
{
    random_matrix(N, 0, 0, t);
}

#define FOREACH_TYPE(T) \
template void random_matrix(stride_type N, matrix<T>& t);
#include "configs/foreach_type.h"

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, const len_vector& len_min, len_vector& len)
{
    auto len_max = random_product_constrained_sequence<len_type>(d, N, len_min);

    len.resize(d);
    for (unsigned i = 0;i < d;i++)
    {
        len[i] = (len_min[i] > 0 ? len_min[i] : random_number<len_type>(1, len_max[i]));
    }
}

matrix<len_type> random_indices(const len_vector& len, double sparsity)
{
    stride_type num_idx = prod(len);
    matrix<len_type> idx({num_idx, (len_type)len.size()});
    stride_type min_idx = (num_idx == 0 ? 0 : 1);

    stride_type i = 0;
    auto it = make_iterator(len);
    while (it.next())
    {
        if (random_number<double>() < sparsity)
        {
            for (unsigned j = 0;j < len.size();j++)
                idx[i][j] = it.position()[j];
            i++;
        }
    }

    return idx[range(max(i,min_idx))];
}

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, varray<T>& A)
{
    len_vector len_A;
    random_lengths(N/sizeof(T), d, len_min, len_A);
    A.reset(len_A);
    randomize_tensor(A);
}

#define FOREACH_TYPE(T) \
template void random_tensor(stride_type N, unsigned d, const len_vector& len_min, varray<T>& A);
#include "configs/foreach_type.h"

template <typename T>
void random_tensor(stride_type N, unsigned d, unsigned nirrep, const len_vector& len_min, dpd_varray<T>& A)
{
    unsigned irrep_A;
    vector<vector<len_type>> len_A(d);

    do
    {
        irrep_A = random_number(nirrep-1);

        len_vector len_A_;
        random_lengths(nirrep*N/sizeof(T), d, len_min, len_A_);

        for (unsigned i = 0;i < d;i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);
    }
    while (dpd_varray<T>::size(irrep_A, len_A) == 0);

    A.reset(irrep_A, nirrep, len_A);
    randomize_tensor(A);
}

#define FOREACH_TYPE(T) \
template void random_tensor(stride_type N, unsigned d, unsigned nirrep, const len_vector& len_min, dpd_varray<T>& A);
#include "configs/foreach_type.h"

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, indexed_varray<T>& A)
{
    len_vector len_A;
    random_lengths(N/sizeof(T), d, len_min, len_A);

    unsigned dense_d = random_number(1u, d);
    auto idxs_A = random_indices(len_vector(len_A.begin()+dense_d, len_A.end()), 0.5);

    A.reset(len_A, idxs_A);
    randomize_tensor(A);
    for (len_type i = 0;i < A.num_indices();i++)
        const_cast<T&>(A.factor(i)) = random_choice({1.0, 0.5, 0.0});
}

#define FOREACH_TYPE(T) \
template void random_tensor(stride_type N, unsigned d, const len_vector& len_min, indexed_varray<T>& A);
#include "configs/foreach_type.h"

template <typename T>
void random_tensor(stride_type N, unsigned d, unsigned nirrep, const len_vector& len_min, indexed_dpd_varray<T>& A)
{
    unsigned irrep_A;
    vector<vector<len_type>> len_A(d);
    len_vector idx_len_A;
    irrep_vector idx_irrep_A;

    do
    {
        irrep_A = random_number(nirrep-1);

        len_vector len_A_;
        random_lengths(nirrep*N/sizeof(T), d, len_min, len_A_);

        for (unsigned i = 0;i < d;i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);
    }
    while (dpd_varray<T>::size(irrep_A, len_A) == 0);

    do
    {
        unsigned dense_d = random_number(1u, d);
        idx_len_A.resize(d-dense_d);
        idx_irrep_A.resize(d-dense_d);
        for (unsigned i = dense_d;i < d;i++)
        {
            idx_irrep_A[i-dense_d] = random_number(nirrep-1);
            idx_len_A[i-dense_d] = len_A[i][idx_irrep_A[i-dense_d]];
        }
    }
    while (prod(idx_len_A) == 0);

    auto idxs_A = random_indices(idx_len_A, 0.1);

    A.reset(irrep_A, nirrep, len_A, idx_irrep_A, idxs_A);
    randomize_tensor(A);
    for (len_type i = 0;i < A.num_indices();i++)
        const_cast<T&>(A.factor(i)) = random_choice({1.0, 0.5, 0.0});
}

#define FOREACH_TYPE(T) \
template void random_tensor(stride_type N, unsigned d, unsigned nirrep, const len_vector& len_min, indexed_dpd_varray<T>& A);
#include "configs/foreach_type.h"

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, dpd_varray<T>& A)
{
    random_tensor(N, d, 1 << random_number(2), len_min, A);
}

#define FOREACH_TYPE(T) \
template void random_tensor(stride_type N, unsigned d, const len_vector& len_min, dpd_varray<T>& A);
#include "configs/foreach_type.h"

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, indexed_dpd_varray<T>& A)
{
    random_tensor(N, d, 1 << random_number(2), len_min, A);
}

#define FOREACH_TYPE(T) \
template void random_tensor(stride_type N, unsigned d, const len_vector& len_min, indexed_dpd_varray<T>& A);
#include "configs/foreach_type.h"

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4 and N/2. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, len_vector& len)
{
    random_lengths(N, d, len_vector(d), len);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
void random_lengths(stride_type N, len_vector& len)
{
    random_lengths(N, random_number(1,8), len);
}

void random_lengths(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    len_vector& len_A, label_vector& idx_A,
                    len_vector& len_B, label_vector& idx_B)
{
    unsigned ndim_A = ndim_A_only+ndim_AB;
    unsigned ndim_B = ndim_B_only+ndim_AB;

    vector<pair<index_type,unsigned>> types_A(ndim_A);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_A_only;j++) types_A[i++] = {TYPE_A, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_A[i++] = {TYPE_AB, j};
    }
    random_shuffle(types_A.begin(), types_A.end());

    vector<pair<index_type,unsigned>> types_B(ndim_B);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_B_only;j++) types_B[i++] = {TYPE_B, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_B[i++] = {TYPE_AB, j};
    }
    random_shuffle(types_B.begin(), types_B.end());

    label_vector idx = range<label_type>('a', static_cast<char>('a'+ndim_A+ndim_B-ndim_AB));
    random_shuffle(idx.begin(), idx.end());

    unsigned c = 0;
    label_vector idx_A_only(ndim_A_only, 0);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    label_vector idx_B_only(ndim_B_only, 0);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    label_vector idx_AB(ndim_AB, 0);
    for (unsigned i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    idx_A.resize(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        switch (types_A[i].first)
        {
            case TYPE_A  : idx_A[i] = idx_A_only[types_A[i].second]; break;
            case TYPE_AB : idx_A[i] = idx_AB    [types_A[i].second]; break;
            default: break;
        }
    }

    idx_B.resize(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        switch (types_B[i].first)
        {
            case TYPE_B  : idx_B[i] = idx_B_only[types_B[i].second]; break;
            case TYPE_AB : idx_B[i] = idx_AB    [types_B[i].second]; break;
            default: break;
        }
    }

    bool switch_AB = ndim_B > ndim_A;

    if (switch_AB)
    {
        swap(ndim_A, ndim_B);
        swap(idx_A, idx_B);
    }

    random_lengths(N, ndim_A, len_A);

    len_vector min_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (idx_B[i] == idx_A[j]) min_B[i] = len_A[j];
        }
    }

    random_lengths(N, ndim_B, min_B, len_B);

    if (switch_AB)
    {
        swap(ndim_A, ndim_B);
        swap(idx_A, idx_B);
        swap(len_A, len_B);
    }
}

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    varray<T>& A, label_vector& idx_A,
                    varray<T>& B, label_vector& idx_B)
{
    len_vector len_A, len_B;

    random_lengths(N/sizeof(T), ndim_A_only, ndim_B_only, ndim_AB,
                   len_A, idx_A, len_B, idx_B);

    A.reset(len_A);
    B.reset(len_B);

    randomize_tensor(A);
    randomize_tensor(B);
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, \
                    unsigned ndim_AB, \
                    varray<T>& A, label_vector& idx_A, \
                    varray<T>& B, label_vector& idx_B);
#include "configs/foreach_type.h"

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    dpd_varray<T>& A, label_vector& idx_A,
                    dpd_varray<T>& B, label_vector& idx_B)
{
    unsigned nirrep;
    unsigned irrep_A, irrep_B;
    vector<vector<len_type>> len_A, len_B;

    do
    {
        nirrep = 1 << random_number(2);
        irrep_A = irrep_B = random_number(nirrep-1);
        if (ndim_A_only || ndim_B_only) irrep_B = random_number(nirrep-1);

        len_vector len_A_, len_B_;
        random_lengths(nirrep*N/sizeof(T), ndim_A_only, ndim_B_only, ndim_AB,
                       len_A_, idx_A, len_B_, idx_B);

        len_A.resize(len_A_.size());
        len_B.resize(len_B_.size());

        for (unsigned i = 0;i < len_A_.size();i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);

        for (unsigned i = 0;i < len_B_.size();i++)
        {
            bool found = false;
            for (unsigned j = 0;j < len_A_.size();j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    len_B[i] = len_A[j];
                    found = true;
                }
            }

            if (!found)
                len_B[i] = random_sum_constrained_sequence<len_type>(nirrep, len_B_[i]);
        }
    }
    while (dpd_varray<T>::size(irrep_A, len_A) == 0 ||
           dpd_varray<T>::size(irrep_B, len_B) == 0);

    A.reset(irrep_A, nirrep, len_A);
    B.reset(irrep_B, nirrep, len_B);
    randomize_tensor(A);
    randomize_tensor(B);
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, \
                    unsigned ndim_AB, \
                    dpd_varray<T>& A, label_vector& idx_A, \
                    dpd_varray<T>& B, label_vector& idx_B);
#include "configs/foreach_type.h"

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    indexed_varray<T>& A, label_vector& idx_A,
                    indexed_varray<T>& B, label_vector& idx_B)
{
    len_vector len_A, len_B;

    random_lengths(N/sizeof(T), ndim_A_only, ndim_B_only, ndim_AB,
                   len_A, idx_A, len_B, idx_B);

    unsigned dense_ndim_A = random_number(1u, ndim_AB+ndim_A_only);
    unsigned dense_ndim_B = random_number(1u, ndim_AB+ndim_B_only);

    auto idxs_A = random_indices(len_vector(len_A.begin()+dense_ndim_A, len_A.end()), 0.5);
    auto idxs_B = random_indices(len_vector(len_B.begin()+dense_ndim_B, len_B.end()), 0.5);

    A.reset(len_A, idxs_A);
    B.reset(len_B, idxs_B);

    randomize_tensor(A);
    randomize_tensor(B);
    for (len_type i = 0;i < A.num_indices();i++)
        const_cast<T&>(A.factor(i)) = random_choice({1.0, 0.5, 0.0});
    for (len_type i = 0;i < B.num_indices();i++)
        const_cast<T&>(B.factor(i)) = random_choice({1.0, 0.5, 0.0});
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, \
                    unsigned ndim_AB, \
                    indexed_varray<T>& A, label_vector& idx_A, \
                    indexed_varray<T>& B, label_vector& idx_B);
#include "configs/foreach_type.h"

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    indexed_dpd_varray<T>& A, label_vector& idx_A,
                    indexed_dpd_varray<T>& B, label_vector& idx_B)
{
    unsigned nirrep;
    unsigned irrep_A, irrep_B;
    vector<vector<len_type>> len_A, len_B;
    len_vector idx_len_A, idx_len_B;
    irrep_vector idx_irrep_A, idx_irrep_B;

    do
    {
        nirrep = 1 << random_number(2);
        irrep_A = irrep_B = random_number(nirrep-1);
        if (ndim_A_only || ndim_B_only) irrep_B = random_number(nirrep-1);

        len_vector len_A_, len_B_;
        random_lengths(nirrep*N/sizeof(T), ndim_A_only, ndim_B_only, ndim_AB,
                       len_A_, idx_A, len_B_, idx_B);

        len_A.resize(len_A_.size());
        len_B.resize(len_B_.size());

        for (unsigned i = 0;i < len_A_.size();i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);

        for (unsigned i = 0;i < len_B_.size();i++)
        {
            bool found = false;
            for (unsigned j = 0;j < len_A_.size();j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    len_B[i] = len_A[j];
                    found = true;
                }
            }

            if (!found)
                len_B[i] = random_sum_constrained_sequence<len_type>(nirrep, len_B_[i]);
        }
    }
    while (dpd_varray<T>::size(irrep_A, len_A) == 0 ||
           dpd_varray<T>::size(irrep_B, len_B) == 0);

    do
    {
        unsigned ndim_A = ndim_AB+ndim_A_only;
        unsigned dense_ndim_A = random_number(1u, ndim_A);
        idx_len_A.resize(ndim_A-dense_ndim_A);
        idx_irrep_A.resize(ndim_A-dense_ndim_A);
        for (unsigned i = dense_ndim_A;i < ndim_A;i++)
        {
            idx_irrep_A[i-dense_ndim_A] = random_number(nirrep-1);
            idx_len_A[i-dense_ndim_A] = len_A[i][idx_irrep_A[i-dense_ndim_A]];
        }

        unsigned ndim_B = ndim_AB+ndim_B_only;
        unsigned dense_ndim_B = random_number(1u, ndim_B);
        idx_len_B.resize(ndim_B-dense_ndim_B);
        idx_irrep_B.resize(ndim_B-dense_ndim_B);
        for (unsigned i = dense_ndim_B;i < ndim_B;i++)
        {
            bool found = false;
            for (unsigned j = dense_ndim_A;j < ndim_A;j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    idx_irrep_B[i-dense_ndim_B] = idx_irrep_A[j-dense_ndim_A];
                    found = true;
                }
            }

            if (!found)
                idx_irrep_B[i-dense_ndim_B] = random_number(nirrep-1);
            idx_len_B[i-dense_ndim_B] = len_B[i][idx_irrep_B[i-dense_ndim_B]];
        }
    }
    while (prod(idx_len_A) == 0 || prod(idx_len_B) == 0);

    auto idxs_A = random_indices(idx_len_A, 0.5);
    auto idxs_B = random_indices(idx_len_B, 0.5);

    A.reset(irrep_A, nirrep, len_A, idx_irrep_A, idxs_A);
    B.reset(irrep_B, nirrep, len_B, idx_irrep_B, idxs_B);
    randomize_tensor(A);
    randomize_tensor(B);
    for (len_type i = 0;i < A.num_indices();i++)
        const_cast<T&>(A.factor(i)) = random_choice({1.0, 0.5, 0.0});
    for (len_type i = 0;i < B.num_indices();i++)
        const_cast<T&>(B.factor(i)) = random_choice({1.0, 0.5, 0.0});
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, \
                    unsigned ndim_AB, \
                    indexed_dpd_varray<T>& A, label_vector& idx_A, \
                    indexed_dpd_varray<T>& B, label_vector& idx_B);
#include "configs/foreach_type.h"

void random_lengths(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    len_vector& len_A, label_vector& idx_A,
                    len_vector& len_B, label_vector& idx_B,
                    len_vector& len_C, label_vector& idx_C)
{
    unsigned ndim_A = ndim_A_only+ndim_AB+ndim_AC+ndim_ABC;
    unsigned ndim_B = ndim_B_only+ndim_AB+ndim_BC+ndim_ABC;
    unsigned ndim_C = ndim_C_only+ndim_AC+ndim_BC+ndim_ABC;

    vector<pair<index_type,unsigned>> types_A(ndim_A);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_A_only;j++) types_A[i++] = {TYPE_A, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_A[i++] = {TYPE_AB, j};
        for (unsigned j = 0;j < ndim_AC    ;j++) types_A[i++] = {TYPE_AC, j};
        for (unsigned j = 0;j < ndim_ABC   ;j++) types_A[i++] = {TYPE_ABC, j};
    }
    random_shuffle(types_A.begin(), types_A.end());

    vector<pair<index_type,unsigned>> types_B(ndim_B);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_B_only;j++) types_B[i++] = {TYPE_B, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_B[i++] = {TYPE_AB, j};
        for (unsigned j = 0;j < ndim_BC    ;j++) types_B[i++] = {TYPE_BC, j};
        for (unsigned j = 0;j < ndim_ABC   ;j++) types_B[i++] = {TYPE_ABC, j};
    }
    random_shuffle(types_B.begin(), types_B.end());

    vector<pair<index_type,unsigned>> types_C(ndim_C);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_C_only;j++) types_C[i++] = {TYPE_C, j};
        for (unsigned j = 0;j < ndim_AC    ;j++) types_C[i++] = {TYPE_AC, j};
        for (unsigned j = 0;j < ndim_BC    ;j++) types_C[i++] = {TYPE_BC, j};
        for (unsigned j = 0;j < ndim_ABC   ;j++) types_C[i++] = {TYPE_ABC, j};
    }
    random_shuffle(types_C.begin(), types_C.end());

    label_vector idx =
        range<label_type>('a', static_cast<char>('a'+ndim_A_only+ndim_B_only+ndim_C_only+
                      ndim_AB+ndim_AC+ndim_BC+ndim_ABC));
    random_shuffle(idx.begin(), idx.end());

    unsigned c = 0;
    label_vector idx_A_only(ndim_A_only, 0);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    label_vector idx_B_only(ndim_B_only, 0);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    label_vector idx_C_only(ndim_C_only, 0);
    for (unsigned i = 0;i < ndim_C_only;i++) idx_C_only[i] = idx[c++];

    label_vector idx_AB(ndim_AB, 0);
    for (unsigned i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    label_vector idx_AC(ndim_AC, 0);
    for (unsigned i = 0;i < ndim_AC;i++) idx_AC[i] = idx[c++];

    label_vector idx_BC(ndim_BC, 0);
    for (unsigned i = 0;i < ndim_BC;i++) idx_BC[i] = idx[c++];

    label_vector idx_ABC(ndim_ABC, 0);
    for (unsigned i = 0;i < ndim_ABC;i++) idx_ABC[i] = idx[c++];

    idx_A.resize(ndim_A);
    for (unsigned i = 0;i < ndim_A;i++)
    {
        switch (types_A[i].first)
        {
            case TYPE_A  : idx_A[i] = idx_A_only[types_A[i].second]; break;
            case TYPE_AB : idx_A[i] = idx_AB    [types_A[i].second]; break;
            case TYPE_AC : idx_A[i] = idx_AC    [types_A[i].second]; break;
            case TYPE_ABC: idx_A[i] = idx_ABC   [types_A[i].second]; break;
            default: break;
        }
    }

    idx_B.resize(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        switch (types_B[i].first)
        {
            case TYPE_B  : idx_B[i] = idx_B_only[types_B[i].second]; break;
            case TYPE_AB : idx_B[i] = idx_AB    [types_B[i].second]; break;
            case TYPE_BC : idx_B[i] = idx_BC    [types_B[i].second]; break;
            case TYPE_ABC: idx_B[i] = idx_ABC   [types_B[i].second]; break;
            default: break;
        }
    }

    idx_C.resize(ndim_C);
    for (unsigned i = 0;i < ndim_C;i++)
    {
        switch (types_C[i].first)
        {
            case TYPE_C  : idx_C[i] = idx_C_only[types_C[i].second]; break;
            case TYPE_AC : idx_C[i] = idx_AC    [types_C[i].second]; break;
            case TYPE_BC : idx_C[i] = idx_BC    [types_C[i].second]; break;
            case TYPE_ABC: idx_C[i] = idx_ABC   [types_C[i].second]; break;
            default: break;
        }
    }

    enum Order {ABC, ACB, BAC, BCA, CAB, CBA};

    Order order;
    if (ndim_A > ndim_B)
    {
        if (ndim_B > ndim_C)
        {
            order = ABC;
        }
        else if (ndim_A > ndim_C)
        {
            order = ACB;
        }
        else
        {
            order = CAB;
        }
    }
    else
    {
        if (ndim_A > ndim_C)
        {
            order = BAC;
        }
        else if (ndim_B > ndim_C)
        {
            order = BCA;
        }
        else
        {
            order = CBA;
        }
    }

    switch (order)
    {
        case ABC: break;
        case ACB:
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            break;
        case BAC:
            swap(ndim_A, ndim_B);
            swap(idx_A, idx_B);
            break;
        case BCA:
            swap(ndim_A, ndim_B);
            swap(idx_A, idx_B);
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            break;
        case CAB:
            swap(ndim_A, ndim_C);
            swap(idx_A, idx_C);
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            break;
        case CBA:
            swap(ndim_A, ndim_C);
            swap(idx_A, idx_C);
            break;
    }

    while (true)
    {
        random_lengths(N, ndim_A, len_A);

        len_vector min_B(ndim_B);
        for (unsigned i = 0;i < ndim_B;i++)
        {
            for (unsigned j = 0;j < ndim_A;j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    min_B[i] = len_A[j];
                    break;
                }
            }
        }

        random_lengths(N, ndim_B, min_B, len_B);

        stride_type siz = 1;
        len_vector min_C(ndim_C);
        for (unsigned i = 0;i < ndim_C;i++)
        {
            bool found = false;
            for (unsigned j = 0;j < ndim_A;j++)
            {
                if (idx_C[i] == idx_A[j])
                {
                    min_C[i] = len_A[j];
                    siz *= min_C[i];
                    found = true;
                    break;
                }
            }
            if (found) continue;
            for (unsigned j = 0;j < ndim_B;j++)
            {
                if (idx_C[i] == idx_B[j])
                {
                    min_C[i] = len_B[j];
                    siz *= min_C[i];
                    break;
                }
            }
        }
        if (siz > N) continue;

        random_lengths(N, ndim_C, min_C, len_C);

        break;
    }

    switch (order)
    {
        case ABC: break;
        case ACB:
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            swap(len_B, len_C);
            break;
        case BAC:
            swap(ndim_A, ndim_B);
            swap(idx_A, idx_B);
            swap(len_A, len_B);
            break;
        case BCA:
            swap(ndim_A, ndim_C);
            swap(idx_A, idx_C);
            swap(len_A, len_C);
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            swap(len_B, len_C);
            break;
        case CAB:
            swap(ndim_A, ndim_B);
            swap(idx_A, idx_B);
            swap(len_A, len_B);
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            swap(len_B, len_C);
            break;
        case CBA:
            swap(ndim_A, ndim_C);
            swap(idx_A, idx_C);
            swap(len_A, len_C);
            break;
    }
}

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    varray<T>& A, label_vector& idx_A,
                    varray<T>& B, label_vector& idx_B,
                    varray<T>& C, label_vector& idx_C)
{
    len_vector len_A, len_B, len_C;

    random_lengths(N/sizeof(T), ndim_A_only, ndim_B_only, ndim_C_only,
                   ndim_AB, ndim_AC, ndim_BC, ndim_ABC,
                   len_A, idx_A, len_B, idx_B, len_C, idx_C);

    A.reset(len_A);
    B.reset(len_B);
    C.reset(len_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only, \
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC, \
                    unsigned ndim_ABC, \
                    varray<T>& A, label_vector& idx_A, \
                    varray<T>& B, label_vector& idx_B, \
                    varray<T>& C, label_vector& idx_C);
#include "configs/foreach_type.h"

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    dpd_varray<T>& A, label_vector& idx_A,
                    dpd_varray<T>& B, label_vector& idx_B,
                    dpd_varray<T>& C, label_vector& idx_C)
{
    unsigned nirrep, irrep_A, irrep_B, irrep_C;
    vector<vector<len_type>> len_A, len_B, len_C;

    do
    {
        nirrep = 1 << random_number(2);
        irrep_A = random_number(nirrep-1);
        irrep_B = random_number(nirrep-1);
        irrep_C = irrep_A^irrep_B;
        if (ndim_A_only || ndim_B_only || ndim_C_only || ndim_ABC)
            irrep_C = random_number(nirrep-1);

        len_vector len_A_, len_B_, len_C_;

        random_lengths(nirrep*N/sizeof(T), ndim_A_only, ndim_B_only, ndim_C_only,
                       ndim_AB, ndim_AC, ndim_BC, ndim_ABC,
                       len_A_, idx_A, len_B_, idx_B, len_C_, idx_C);

        len_A.resize(len_A_.size());
        len_B.resize(len_B_.size());
        len_C.resize(len_C_.size());

        for (unsigned i = 0;i < len_A_.size();i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);

        for (unsigned i = 0;i < len_B_.size();i++)
        {
            bool found = false;
            for (unsigned j = 0;j < len_A_.size();j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    len_B[i] = len_A[j];
                    found = true;
                }
            }

            if (!found)
                len_B[i] = random_sum_constrained_sequence<len_type>(nirrep, len_B_[i]);
        }

        for (unsigned i = 0;i < len_C_.size();i++)
        {
            bool found = false;
            for (unsigned j = 0;j < len_A_.size();j++)
            {
                if (idx_C[i] == idx_A[j])
                {
                    len_C[i] = len_A[j];
                    found = true;
                }
            }

            for (unsigned j = 0;j < len_B_.size();j++)
            {
                if (idx_C[i] == idx_B[j])
                {
                    len_C[i] = len_B[j];
                    found = true;
                }
            }

            if (!found)
                len_C[i] = random_sum_constrained_sequence<len_type>(nirrep, len_C_[i]);
        }
    }
    while (dpd_varray<T>::size(irrep_A, len_A) == 0 ||
           dpd_varray<T>::size(irrep_B, len_B) == 0 ||
           dpd_varray<T>::size(irrep_C, len_C) == 0);

    A.reset(irrep_A, nirrep, len_A);
    B.reset(irrep_B, nirrep, len_B);
    C.reset(irrep_C, nirrep, len_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only, \
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC, \
                    unsigned ndim_ABC, \
                    dpd_varray<T>& A, label_vector& idx_A, \
                    dpd_varray<T>& B, label_vector& idx_B, \
                    dpd_varray<T>& C, label_vector& idx_C);
#include "configs/foreach_type.h"

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    indexed_varray<T>& A, label_vector& idx_A,
                    indexed_varray<T>& B, label_vector& idx_B,
                    indexed_varray<T>& C, label_vector& idx_C)
{
    len_vector len_A, len_B, len_C;

    random_lengths(N/sizeof(T), ndim_A_only, ndim_B_only, ndim_C_only,
                   ndim_AB, ndim_AC, ndim_BC, ndim_ABC,
                   len_A, idx_A, len_B, idx_B, len_C, idx_C);

    unsigned dense_ndim_A = random_number(1u, ndim_ABC+ndim_AB+ndim_AC+ndim_A_only);
    unsigned dense_ndim_B = random_number(1u, ndim_ABC+ndim_AB+ndim_BC+ndim_B_only);
    unsigned dense_ndim_C = random_number(1u, ndim_ABC+ndim_AC+ndim_BC+ndim_C_only);

    auto idxs_A = random_indices(len_vector(len_A.begin()+dense_ndim_A, len_A.end()), 0.5);
    auto idxs_B = random_indices(len_vector(len_B.begin()+dense_ndim_B, len_B.end()), 0.5);
    auto idxs_C = random_indices(len_vector(len_C.begin()+dense_ndim_C, len_C.end()), 0.5);

    A.reset(len_A, idxs_A);
    B.reset(len_B, idxs_B);
    C.reset(len_C, idxs_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
    for (len_type i = 0;i < A.num_indices();i++)
        const_cast<T&>(A.factor(i)) = random_choice({1.0, 0.5, 0.0});
    for (len_type i = 0;i < B.num_indices();i++)
        const_cast<T&>(B.factor(i)) = random_choice({1.0, 0.5, 0.0});
    for (len_type i = 0;i < C.num_indices();i++)
        const_cast<T&>(C.factor(i)) = random_choice({1.0, 0.5, 0.0});
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only, \
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC, \
                    unsigned ndim_ABC, \
                    indexed_varray<T>& A, label_vector& idx_A, \
                    indexed_varray<T>& B, label_vector& idx_B, \
                    indexed_varray<T>& C, label_vector& idx_C);
#include "configs/foreach_type.h"

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    indexed_dpd_varray<T>& A, label_vector& idx_A,
                    indexed_dpd_varray<T>& B, label_vector& idx_B,
                    indexed_dpd_varray<T>& C, label_vector& idx_C)
{
    unsigned nirrep, irrep_A, irrep_B, irrep_C;
    vector<vector<len_type>> len_A, len_B, len_C;
    len_vector idx_len_A, idx_len_B, idx_len_C;
    irrep_vector idx_irrep_A, idx_irrep_B, idx_irrep_C;

    do
    {
        nirrep = 1 << random_number(2);
        irrep_A = random_number(nirrep-1);
        irrep_B = random_number(nirrep-1);
        irrep_C = irrep_A^irrep_B;
        if (ndim_A_only || ndim_B_only || ndim_C_only || ndim_ABC)
            irrep_C = random_number(nirrep-1);

        len_vector len_A_, len_B_, len_C_;

        random_lengths(nirrep*N/sizeof(T), ndim_A_only, ndim_B_only, ndim_C_only,
                       ndim_AB, ndim_AC, ndim_BC, ndim_ABC,
                       len_A_, idx_A, len_B_, idx_B, len_C_, idx_C);

        len_A.resize(len_A_.size());
        len_B.resize(len_B_.size());
        len_C.resize(len_C_.size());

        for (unsigned i = 0;i < len_A_.size();i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);

        for (unsigned i = 0;i < len_B_.size();i++)
        {
            bool found = false;
            for (unsigned j = 0;j < len_A_.size();j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    len_B[i] = len_A[j];
                    found = true;
                }
            }

            if (!found)
                len_B[i] = random_sum_constrained_sequence<len_type>(nirrep, len_B_[i]);
        }

        for (unsigned i = 0;i < len_C_.size();i++)
        {
            bool found = false;
            for (unsigned j = 0;j < len_A_.size();j++)
            {
                if (idx_C[i] == idx_A[j])
                {
                    len_C[i] = len_A[j];
                    found = true;
                }
            }

            for (unsigned j = 0;j < len_B_.size();j++)
            {
                if (idx_C[i] == idx_B[j])
                {
                    len_C[i] = len_B[j];
                    found = true;
                }
            }

            if (!found)
                len_C[i] = random_sum_constrained_sequence<len_type>(nirrep, len_C_[i]);
        }
    }
    while (dpd_varray<T>::size(irrep_A, len_A) == 0 ||
           dpd_varray<T>::size(irrep_B, len_B) == 0 ||
           dpd_varray<T>::size(irrep_C, len_C) == 0);

    do
    {
        unsigned ndim_A = ndim_ABC+ndim_AB+ndim_AC+ndim_A_only;
        unsigned dense_ndim_A = random_number(1u, ndim_A);
        idx_len_A.resize(ndim_A-dense_ndim_A);
        idx_irrep_A.resize(ndim_A-dense_ndim_A);
        for (unsigned i = dense_ndim_A;i < ndim_A;i++)
        {
            idx_irrep_A[i-dense_ndim_A] = random_number(nirrep-1);
            idx_len_A[i-dense_ndim_A] = len_A[i][idx_irrep_A[i-dense_ndim_A]];
        }

        unsigned ndim_B = ndim_ABC+ndim_AB+ndim_BC+ndim_B_only;
        unsigned dense_ndim_B = random_number(1u, ndim_B);
        idx_len_B.resize(ndim_B-dense_ndim_B);
        idx_irrep_B.resize(ndim_B-dense_ndim_B);
        for (unsigned i = dense_ndim_B;i < ndim_B;i++)
        {
            bool found = false;
            for (unsigned j = dense_ndim_A;j < ndim_A;j++)
            {
                if (idx_B[i] == idx_A[j])
                {
                    idx_irrep_B[i-dense_ndim_B] = idx_irrep_A[j-dense_ndim_A];
                    found = true;
                }
            }

            if (!found)
                idx_irrep_B[i-dense_ndim_B] = random_number(nirrep-1);
            idx_len_B[i-dense_ndim_B] = len_B[i][idx_irrep_B[i-dense_ndim_B]];
        }

        unsigned ndim_C = ndim_ABC+ndim_AC+ndim_BC+ndim_C_only;
        unsigned dense_ndim_C = random_number(1u, ndim_C);
        idx_len_C.resize(ndim_C-dense_ndim_C);
        idx_irrep_C.resize(ndim_C-dense_ndim_C);
        for (unsigned i = dense_ndim_C;i < ndim_C;i++)
        {
            bool found = false;
            for (unsigned j = dense_ndim_A;j < ndim_A;j++)
            {
                if (idx_C[i] == idx_A[j])
                {
                    idx_irrep_C[i-dense_ndim_C] = idx_irrep_A[j-dense_ndim_A];
                    found = true;
                }
            }
            for (unsigned j = dense_ndim_B;j < ndim_B;j++)
            {
                if (idx_C[i] == idx_B[j])
                {
                    idx_irrep_C[i-dense_ndim_C] = idx_irrep_B[j-dense_ndim_B];
                    found = true;
                }
            }

            if (!found)
                idx_irrep_C[i-dense_ndim_C] = random_number(nirrep-1);
            idx_len_C[i-dense_ndim_C] = len_C[i][idx_irrep_C[i-dense_ndim_C]];
        }
    }
    while (prod(idx_len_A) == 0 || prod(idx_len_B) == 0 || prod(idx_len_C) == 0);

    auto idxs_A = random_indices(idx_len_A, 0.5);
    auto idxs_B = random_indices(idx_len_B, 0.5);
    auto idxs_C = random_indices(idx_len_C, 0.5);

    A.reset(irrep_A, nirrep, len_A, idx_irrep_A, idxs_A);
    B.reset(irrep_B, nirrep, len_B, idx_irrep_B, idxs_B);
    C.reset(irrep_C, nirrep, len_C, idx_irrep_C, idxs_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
    for (len_type i = 0;i < A.num_indices();i++)
        const_cast<T&>(A.factor(i)) = random_choice({1.0, 0.5, 0.0});
    for (len_type i = 0;i < B.num_indices();i++)
        const_cast<T&>(B.factor(i)) = random_choice({1.0, 0.5, 0.0});
    for (len_type i = 0;i < C.num_indices();i++)
        const_cast<T&>(C.factor(i)) = random_choice({1.0, 0.5, 0.0});
}

#define FOREACH_TYPE(T) \
template \
void random_tensors(stride_type N, \
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only, \
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC, \
                    unsigned ndim_ABC, \
                    indexed_dpd_varray<T>& A, label_vector& idx_A, \
                    indexed_dpd_varray<T>& B, label_vector& idx_B, \
                    indexed_dpd_varray<T>& C, label_vector& idx_C);
#include "configs/foreach_type.h"

int main(int argc, char **argv)
{
    time_t seed = chrono::duration_cast<chrono::nanoseconds>(
        chrono::high_resolution_clock::now().time_since_epoch()).count();

    struct option opts[] = {{"size", required_argument, NULL, 'n'},
                            {"rep",  required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {"filter", required_argument, NULL, 'f'},
                            {0, 0, 0, 0}};

    vector<const char*> catch_argv = {"tblis::test", "-d", "yes", "-a"};
    string extra_args;

    int arg;
    int index;
    while ((arg = getopt_long(argc, argv, "n:r:s:v", opts, &index)) != -1)
    {
        istringstream iss;
        switch (arg)
        {
            case 'n':
                iss.str(optarg);
                iss >> N;
                break;
            case 'r':
                iss.str(optarg);
                iss >> R;
                break;
            case 's':
                iss.str(optarg);
                iss >> seed;
                break;
            case 'v':
                catch_argv.push_back("-s");
                break;
            case 'f':
                extra_args = optarg;
                catch_argv.push_back(extra_args.c_str());
                break;
            case '?':
                ::abort();
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    rand_engine.seed(seed);

    cout << "Running tests with " << tblis_get_num_threads() << " threads\n";
    cout << endl;

    int nfailed = Catch::Session().run(catch_argv.size(), catch_argv.data());

    return nfailed ? 1 : 0;
}
