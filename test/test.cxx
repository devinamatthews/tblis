#include <algorithm>
#include <limits>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <iomanip>

#include "tblis.h"

#include "internal/3t/mult.hpp"
#include "util/random.hpp"
#include "external/stl_ext/include/iostream.hpp"

using namespace std;
using namespace tblis;
using namespace tblis::internal;

template <typename T>
void gemm_ref(T alpha, const_matrix_view<T> A,
                       const_matrix_view<T> B,
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

static vector<unsigned> permutation(unsigned ndim, const label_type* from, const label_type* to)
{
    vector<unsigned> p(ndim);

    for (unsigned i = 0;i < ndim;i++)
    {
        for (unsigned j = 0;j < ndim;j++)
        {
            if (from[j] == to[i])
            {
                p[i] = j;
                break;
            }
        }
    }

    return p;
}

template <typename T, typename U>
void passfail(const string& label, stride_type ia, stride_type ib, T a, U b)
{
    auto c = std::abs(a-b)/(std::abs((a+b)/U(2.0)+U(1e-15)));
    bool pass = (sizeof(c) == 4 ? c < 1e-3 : c < 1e-11) && ia == ib;

    cout << label << ": ";
    if (pass)
    {
        cout << "pass" << endl;
    }
    else
    {
        cout << "fail" << endl;
        cout << std::scientific << std::setprecision(15);
        cout << a << " " << ia << endl;
        cout << b << " " << ib << endl;
        cout << c << endl;
        ::abort();
    }
}

template <typename T, typename U>
void passfail(const string& label, T a, U b)
{
    passfail(label, 0, 0, a, b);
}

template <typename T> const string& type_name();

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
    vector<len_type> len = random_product_constrained_sequence<len_type>(2, N, {m_min, n_min});

    len_type m = (m_min > 0 ? m_min : random_number<len_type>(1, len[0]));
    len_type n = (n_min > 0 ? n_min : random_number<len_type>(1, len[1]));

    t.reset({m, n});

    T* data = t.data();
    MArray::miterator<2> it(t.lengths(), t.strides());
    while (it.next(data)) *data = random_unit<T>();
}

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

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void random_tensor(stride_type N, unsigned d, vector<len_type> len_min, tensor<T>& t)
{
    vector<len_type> len_max = random_product_constrained_sequence<len_type>(d, N, len_min);

    vector<len_type> len(d);
    for (unsigned i = 0;i < d;i++)
    {
        len[i] = (len_min[i] > 0 ? len_min[i] : random_number<len_type>(1, len_max[i]));
    }

    t.reset(len);

    T* data = t.data();
    MArray::viterator<> it(t.lengths(), t.strides());
    while (it.next(data)) *data = random_unit<T>();
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4 and N/2. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void random_tensor(stride_type N, unsigned d, tensor<T>& t)
{
    random_tensor(N, d, vector<len_type>(d), t);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
template <typename T>
void random_tensor(stride_type N, tensor<T>& t)
{
    random_tensor(N, random_number(1,8), t);
}

enum index_type
{
    TYPE_A,
    TYPE_B,
    TYPE_C,
    TYPE_AB,
    TYPE_AC,
    TYPE_BC,
    TYPE_ABC
};

template <typename T>
void random_tensors(stride_type N,
                   unsigned ndim_A_only, unsigned ndim_B_only,
                   unsigned ndim_AB,
                   tensor<T>& A, std::vector<label_type>& idx_A,
                   tensor<T>& B, std::vector<label_type>& idx_B)
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

    vector<label_type> idx = range<label_type>('a', static_cast<char>('a'+ndim_A+ndim_B-ndim_AB));
    random_shuffle(idx.begin(), idx.end());

    unsigned c = 0;
    vector<label_type> idx_A_only(ndim_A_only);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    vector<label_type> idx_B_only(ndim_B_only);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    vector<label_type> idx_AB(ndim_AB);
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

    random_tensor(N, ndim_A, A);

    vector<len_type> min_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (idx_B[i] == idx_A[j]) min_B[i] = A.length(j);
        }
    }

    random_tensor(N, ndim_B, min_B, B);

    if (switch_AB)
    {
        swap(ndim_A, ndim_B);
        swap(idx_A, idx_B);
        swap(A, B);
    }
}

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    tensor<T>& A, std::vector<label_type>& idx_A,
                    tensor<T>& B, std::vector<label_type>& idx_B,
                    tensor<T>& C, std::vector<label_type>& idx_C)
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

    vector<label_type> idx =
        MArray::range<label_type>('a', static_cast<char>('a'+ndim_A_only+ndim_B_only+ndim_C_only+
                      ndim_AB+ndim_AC+ndim_BC+ndim_ABC));
    random_shuffle(idx.begin(), idx.end());

    unsigned c = 0;
    vector<label_type> idx_A_only(ndim_A_only);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    vector<label_type> idx_B_only(ndim_B_only);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    vector<label_type> idx_C_only(ndim_C_only);
    for (unsigned i = 0;i < ndim_C_only;i++) idx_C_only[i] = idx[c++];

    vector<label_type> idx_AB(ndim_AB);
    for (unsigned i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    vector<label_type> idx_AC(ndim_AC);
    for (unsigned i = 0;i < ndim_AC;i++) idx_AC[i] = idx[c++];

    vector<label_type> idx_BC(ndim_BC);
    for (unsigned i = 0;i < ndim_BC;i++) idx_BC[i] = idx[c++];

    vector<label_type> idx_ABC(ndim_ABC);
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
        random_tensor(N, ndim_A, A);

        vector<len_type> min_B(ndim_B);
        for (unsigned i = 0;i < ndim_B;i++)
        {
            for (unsigned j = 0;j < ndim_A;j++)
            {
                if (idx_B[i] == idx_A[j]) min_B[i] = A.length(j);
            }
        }

        random_tensor(N, ndim_B, min_B, B);

        stride_type siz = 1;
        vector<len_type> min_C(ndim_C);
        for (unsigned i = 0;i < ndim_C;i++)
        {
            for (unsigned j = 0;j < ndim_A;j++)
            {
                if (idx_C[i] == idx_A[j])
                {
                    min_C[i] = A.length(j);
                    siz *= min_C[i];
                }
            }
            for (unsigned j = 0;j < ndim_B;j++)
            {
                if (idx_C[i] == idx_B[j])
                {
                    min_C[i] = B.length(j);
                    siz *= min_C[i];
                }
            }
        }
        if (siz > N) continue;

        random_tensor(N, ndim_C, min_C, C);

        break;
    }

    switch (order)
    {
        case ABC: break;
        case ACB:
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            swap(B, C);
            break;
        case BAC:
            swap(ndim_A, ndim_B);
            swap(idx_A, idx_B);
            swap(A, B);
            break;
        case BCA:
            swap(ndim_A, ndim_C);
            swap(idx_A, idx_C);
            swap(A, C);
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            swap(B, C);
            break;
        case CAB:
            swap(ndim_A, ndim_B);
            swap(idx_A, idx_B);
            swap(A, B);
            swap(ndim_B, ndim_C);
            swap(idx_B, idx_C);
            swap(B, C);
            break;
        case CBA:
            swap(ndim_A, ndim_C);
            swap(idx_A, idx_C);
            swap(A, C);
            break;
    }
}

/*
 * Creates a random tensor addmation operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_add(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                          tensor<T>& B, std::vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);
    unsigned ndim_B = random_number(1,8);

    unsigned ndim_AB = random_number(0u, min(ndim_A,ndim_B));
    unsigned ndim_A_only = ndim_A-ndim_AB;
    unsigned ndim_B_only = ndim_B-ndim_AB;

    random_tensors(N,
                   ndim_A_only, ndim_B_only,
                   ndim_AB,
                   A, idx_A,
                   B, idx_B);
}

/*
 * Creates a random tensor trace operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_trace(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                            tensor<T>& B, std::vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);
    unsigned ndim_B = random_number(1,8);

    if (ndim_A < ndim_B) swap(ndim_A, ndim_B);

    random_tensors(N,
                  ndim_A-ndim_B, 0,
                  ndim_B,
                  A, idx_A,
                  B, idx_B);
}

/*
 * Creates a random tensor replication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_replicate(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                                tensor<T>& B, std::vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);
    unsigned ndim_B = random_number(1,8);

    if (ndim_B < ndim_A) swap(ndim_A, ndim_B);

    random_tensors(N,
                   0, ndim_B-ndim_A,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

/*
 * Creates a random tensor transpose operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_transpose(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                                tensor<T>& B, std::vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);

    random_tensors(N,
                  0, 0,
                  ndim_A,
                  A, idx_A,
                  B, idx_B);
}

/*
 * Creates a random tensor dot product operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_dot(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                          tensor<T>& B, std::vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);

    random_tensors(N,
                  0, 0,
                  ndim_A,
                  A, idx_A,
                  B, idx_B);
}

/*
 * Creates a random tensor multiplication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_mult(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                           tensor<T>& B, std::vector<label_type>& idx_B,
                           tensor<T>& C, std::vector<label_type>& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_A_only, ndim_B_only, ndim_C_only;
    int ndim_AB, ndim_AC, ndim_BC;
    int ndim_ABC;
    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = random_number(1,8);
        ndim_A_only = random_number(    ndim_A);
        ndim_B_only = random_number(    ndim_B);
        ndim_C_only = random_number(    ndim_C);
        ndim_ABC    = random_number(min(ndim_A,
                                    min(ndim_B,
                                        ndim_C)));
        ndim_AB     = ((ndim_A-ndim_A_only)+
                       (ndim_B-ndim_B_only)-
                       (ndim_C-ndim_C_only)-ndim_ABC)/2;
        ndim_AC = ndim_A-ndim_A_only-ndim_ABC-ndim_AB;
        ndim_BC = ndim_B-ndim_B_only-ndim_ABC-ndim_AB;
    }
    while (ndim_AB < 0 ||
           ndim_AC < 0 ||
           ndim_BC < 0 ||
           ((ndim_A-ndim_A_only)+
            (ndim_B-ndim_B_only)-
            (ndim_C-ndim_C_only)-ndim_ABC)%2 != 0);

    random_tensors(N,
                  ndim_A_only, ndim_B_only, ndim_C_only,
                  ndim_AB, ndim_AC, ndim_BC,
                  ndim_ABC,
                  A, idx_A,
                  B, idx_B,
                  C, idx_C);
}

/*
 * Creates a random matrix multiplication operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_gemm(stride_type N, matrix<T>& A,
                           matrix<T>& B,
                           matrix<T>& C)
{
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N))));
    len_type n = random_number<len_type>(1, lrint(floor(sqrt(N))));
    len_type k = random_number<len_type>(1, lrint(floor(sqrt(N))));

    //m += (MR<T>::value-1)-(m-1)%MR<T>::value;
    //n += (NR<T>::value-1)-(n-1)%NR<T>::value;
    //k += (KR<T>::value-1)-(k-1)%KR<T>::value;

    //m = 46;
    //n = 334;
    //k = 28;

    //engine.seed(0);

    random_matrix(N, m, k, A);
    random_matrix(N, k, n, B);
    random_matrix(N, m, n, C);

    //printf("%.15f %.15f\n", (double)real(tblis_normfm(A)), (double)real(tblis_normfm(B)));
}

/*
 * Creates a random matrix times vector operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_gemv(stride_type N, matrix<T>& A,
                           matrix<T>& B,
                           matrix<T>& C)
{
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N))));
    len_type k = random_number<len_type>(1, lrint(floor(sqrt(N))));

    random_matrix(N, m, k, A);
    random_matrix(N, k, 1, B);
    random_matrix(N, m, 1, C);
}

/*
 * Creates a random matrix outer product operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_ger(stride_type N, matrix<T>& A,
                          matrix<T>& B,
                          matrix<T>& C)
{
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N))));
    len_type n = random_number<len_type>(1, lrint(floor(sqrt(N))));

    random_matrix(N, m, 1, A);
    random_matrix(N, 1, n, B);
    random_matrix(N, m, n, C);
}

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_contract(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                              tensor<T>& B, std::vector<label_type>& idx_B,
                              tensor<T>& C, std::vector<label_type>& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AB, ndim_AC, ndim_BC;
    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = random_number(1,8);
        ndim_AB = (ndim_A+ndim_B-ndim_C)/2;
        ndim_AC = ndim_A-ndim_AB;
        ndim_BC = ndim_B-ndim_AB;
    }
    while (ndim_AB < 0 ||
           ndim_AC < 0 ||
           ndim_BC < 0 ||
           (ndim_A+ndim_B+ndim_C)%2 != 0);

    random_tensors(N,
                  0, 0, 0,
                  ndim_AB, ndim_AC, ndim_BC,
                  0,
                  A, idx_A,
                  B, idx_B,
                  C, idx_C);
}

/*
 * Creates a random tensor weighting operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_weight(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                            tensor<T>& B, std::vector<label_type>& idx_B,
                            tensor<T>& C, std::vector<label_type>& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AC, ndim_BC;
    int ndim_ABC;
    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = random_number(1,8);
        ndim_ABC = ndim_A+ndim_B-ndim_C;
        ndim_AC = ndim_A-ndim_ABC;
        ndim_BC = ndim_B-ndim_ABC;
    }
    while (ndim_AC  < 0 ||
           ndim_BC  < 0 ||
           ndim_ABC < 0);

    random_tensors(N,
                  0, 0, 0,
                  0, ndim_AC, ndim_BC,
                  ndim_ABC,
                  A, idx_A,
                  B, idx_B,
                  C, idx_C);
}

/*
 * Creates a random tensor outer product operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_outer_prod(stride_type N, tensor<T>& A, std::vector<label_type>& idx_A,
                                 tensor<T>& B, std::vector<label_type>& idx_B,
                                 tensor<T>& C, std::vector<label_type>& idx_C)
{
    unsigned ndim_A, ndim_B, ndim_C;
    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = ndim_A+ndim_B;
    }
    while (ndim_C > 8);

    random_tensors(N,
                  0, 0, 0,
                  0, ndim_A, ndim_B,
                  0,
                  A, idx_A,
                  B, idx_B,
                  C, idx_C);
}

template <typename T>
void test_tblis(stride_type N)
{
    matrix<T> A, B, C, D;

    for (int pass = 0;pass < 3;pass++)
    {
        switch (pass)
        {
            case 0: random_gemm(N, A, B, C); break;
            case 1: random_gemv(N, A, B, C); break;
            case 2: random_ger (N, A, B, C); break;
        }

        D.reset(C);

        T scale = 10.0*random_unit<T>();

        cout << endl;
        cout << "Testing TBLIS/" << (pass == 0 ? "GEMM" :
                                     pass == 1 ? "GEMV" :
                                                 "GER") << " (" << type_name<T>() << "):" << endl;

        cout << endl;
        cout << "m, n, k    = " << C.length(0) << ", " << C.length(1) << ", " << A.length(1) << endl;
        cout << "rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1) << endl;
        cout << "rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1) << endl;
        cout << "rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1) << endl;
        cout << endl;

        D = C;
        gemm_ref(scale, A, B, scale, D);
        T ref_val = reduce(REDUCE_NORM_2, D).first;

        D = C;
        mult(scale, A, B, scale, D);
        T calc_val = reduce(REDUCE_NORM_2, D).first;

        passfail("REF", ref_val, calc_val);
    }
}

template <typename T>
void test_mult(stride_type N)
{
    tensor<T> A, B, C, D;
    std::vector<label_type> idx_A, idx_B, idx_C;

    T scale = 10.0*random_unit<T>();

    cout << endl;
    cout << "Testing mult (" << type_name<T>() << "):" << endl;

    random_mult(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.lengths() << endl;
    cout << "stride_C = " << C.strides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    impl = BLAS_BASED;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T ref_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    impl = REFERENCE;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T calc_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void test_contract(stride_type N)
{
    tensor<T> A, B, C, D;
    std::vector<label_type> idx_A, idx_B, idx_C;

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    T scale = 10.0*random_unit<T>();

    cout << endl;
    cout << "Testing contract (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.lengths() << endl;
    cout << "stride_C = " << C.strides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    impl = REFERENCE;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T ref_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    impl = BLAS_BASED;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T calc_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    passfail("BLAS", ref_val, calc_val);

    impl = BLIS_BASED;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    calc_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    passfail("BLIS", ref_val, calc_val);
}

template <typename T>
void test_weight(stride_type N)
{
    tensor<T> A, B, C, D;
    std::vector<label_type> idx_A, idx_B, idx_C;

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing weight (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.lengths() << endl;
    cout << "stride_C = " << C.strides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    T scale = 10.0*random_unit<T>();

    impl = BLAS_BASED;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T ref_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    impl = REFERENCE;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T calc_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void test_outer_prod(stride_type N)
{
    tensor<T> A, B, C, D;
    std::vector<label_type> idx_A, idx_B, idx_C;

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing outer prod (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.lengths() << endl;
    cout << "stride_C = " << C.strides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    T scale = 10.0*random_unit<T>();

    impl = BLAS_BASED;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T ref_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    impl = REFERENCE;
    D.reset(C);
    mult(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());
    T calc_val = reduce(REDUCE_NORM_2, D, idx_C.data()).first;

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void test_add(stride_type N)
{
    tensor<T> A, B, C;
    std::vector<label_type> idx_A, idx_B;

    T scale = 10.0*random_unit<T>();

    cout << endl;
    cout << "Testing add (" << type_name<T>() << "):" << endl;

    random_add(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    stride_type sz = 1;
    for (unsigned i = 0;i < B.dimension();i++)
    {
        bool found = false;
        for (unsigned j = 0;j < A.dimension();j++)
        {
            if (idx_A[j] == idx_B[i])
            {
                found = true;
                break;
            }
        }
        if (!found) sz *= B.length(i);
    }

    T ref_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce(REDUCE_SUM, B, idx_B.data()).first;
    add(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(sz*ref_val+add_b), calc_val);
}

template <typename T>
void test_trace(stride_type N)
{
    tensor<T> A, B;
    std::vector<label_type> idx_A, idx_B;

    random_trace(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing trace (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    T scale = 10.0*random_unit<T>();

    T ref_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce(REDUCE_SUM, B, idx_B.data()).first;
    add(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(ref_val+add_b), calc_val);
}

template <typename T>
void test_replicate(stride_type N)
{
    tensor<T> A, B;
    std::vector<label_type> idx_A, idx_B;

    random_replicate(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing replicate (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    T scale = 10.0*random_unit<T>();

    stride_type sz = 1;
    for (unsigned i = 0;i < B.dimension();i++)
    {
        bool found = false;
        for (unsigned j = 0;j < A.dimension();j++)
        {
            if (idx_A[j] == idx_B[i])
            {
                found = true;
                break;
            }
        }
        if (!found) sz *= B.length(i);
    }

    T ref_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce(REDUCE_SUM, B, idx_B.data()).first;
    add(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(sz*ref_val+add_b), calc_val);

    ref_val = reduce(REDUCE_NORM_1, A, idx_A.data()).first;
    add(scale, A, idx_A.data(), T(0.0), B, idx_B.data());
    calc_val = reduce(REDUCE_NORM_1, B, idx_B.data()).first;
    passfail("NRM1", sz*T(std::abs(scale))*ref_val, calc_val);
}

template <typename T>
void test_dot(stride_type N)
{
    tensor<T> A, B;
    std::vector<label_type> idx_A, idx_B;

    random_dot(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing dot (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    add(T(1.0), A, idx_A.data(), T(0.0), B, idx_B.data());
    T* data = B.data();
    MArray::viterator<> it(B.lengths(), B.strides());
    while (it.next(data)) *data = stl_ext::conj(*data);
    T ref_val = reduce(REDUCE_NORM_2, A, idx_A.data()).first;
    T calc_val = dot(A, idx_A.data(), B, idx_B.data());
    passfail("NRM2", ref_val*ref_val, calc_val);

    B = T(1);
    ref_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    calc_val = dot(A, idx_A.data(), B, idx_B.data());
    passfail("UNIT", ref_val, calc_val);

    B = T(0);
    calc_val = dot(A, idx_A.data(), B, idx_B.data());
    passfail("ZERO", T(0), calc_val);
}

template <typename T>
void test_transpose(stride_type N)
{
    tensor<T> A, B, C;
    std::vector<label_type> idx_A, idx_B;

    random_transpose(N, A, idx_A, B, idx_B);

    unsigned ndim = A.dimension();
    vector<unsigned> perm = permutation(ndim, idx_A.data(), idx_B.data());

    cout << endl;
    cout << "Testing transpose (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << "perm   = " << perm << endl;
    cout << endl;

    T scale = 10.0*random_unit<T>();

    C.reset(A);
    T ref_val = reduce(REDUCE_NORM_2, A, idx_A.data()).first;
    add(T(1), A, idx_A.data(), T(0), B, idx_B.data());
    add(scale, B, idx_B.data(), scale, C, idx_A.data());
    T calc_val = reduce(REDUCE_NORM_2, C, idx_A.data()).first;
    passfail("INVERSE", T(2.0*std::abs(scale))*ref_val, calc_val);

    B.reset(A);
    idx_B = idx_A;
    vector<label_type> idx_C(ndim);
    vector<len_type> len_C(ndim);
    do
    {
        for (unsigned i = 0;i < ndim;i++)
        {
            unsigned j; for (j = 0;j < ndim && idx_A[j] != static_cast<label_type>(perm[i]+'a');j++) continue;
            idx_C[i] = idx_B[j];
            len_C[i] = B.length(j);
        }
        C.reset(len_C);
        add(T(1), B, idx_B.data(), T(0), C, idx_C.data());
        B.reset(C);
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    calc_val = reduce(REDUCE_NORM_2, C, idx_C.data()).first;
    passfail("CYCLE", ref_val, calc_val);
}

template <typename T>
void test_scale(stride_type N)
{
    tensor<T> A;

    random_tensor(N, A);
    std::vector<label_type> idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    cout << endl;
    cout << "Testing scale (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    T ref_val = reduce(REDUCE_SUM, A, idx_A.data()).first;

    T scale = 10.0*random_unit<T>();
    tblis::scale(scale, A, idx_A.data());
    T calc_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    passfail("RANDOM", ref_val*scale, calc_val);

    tblis::scale(T(1.0), A, idx_A.data());
    calc_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    passfail("UNIT", ref_val*scale, calc_val);

    tblis::scale(T(0.0), A, idx_A.data());
    calc_val = reduce(REDUCE_SUM, A, idx_A.data()).first;
    passfail("ZERO", T(0), calc_val);
}

template <typename T>
void test_reduce(stride_type N)
{
    tensor<T> A;

    random_tensor(N, A);
    stride_type NA = A.stride(A.dimension()-1)*A.length(A.dimension()-1);
    std::vector<label_type> idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    cout << endl;
    cout << "Testing reduction (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

    reduce(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        blas_val += data[i];
    }
    passfail("REDUCE_SUM", ref_val, blas_val);

    reduce(REDUCE_SUM_ABS, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        blas_val += std::abs(data[i]);
    }
    passfail("REDUCE_SUM_ABS", ref_val, blas_val);

    reduce(REDUCE_MAX, A, idx_A.data(), ref_val, ref_idx);
    blas_val = data[0];
    blas_idx = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        if (data[i] > blas_val)
        {
            blas_val = data[i];
            blas_idx = i;
        }
    }
    passfail("REDUCE_MAX", ref_idx, blas_idx, ref_val, blas_val);

    reduce(REDUCE_MAX_ABS, A, idx_A.data(), ref_val, ref_idx);
    blas_val = std::abs(data[0]);
    blas_idx = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        if (std::abs(data[i]) > blas_val)
        {
            blas_val = std::abs(data[i]);
            blas_idx = i;
        }
    }
    passfail("REDUCE_MAX_ABS", ref_idx, blas_idx, ref_val, blas_val);

    /*
    reduce(REDUCE_MIN, A, idx_A.data(), ref_val, ref_idx);
    set(data[0], blas_val);
    blas_idx = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        if (data[i] < blas_val)
        {
            set(data[i], blas_val);
            blas_idx = i;
        }
    }
    passfail("REDUCE_MIN", ref_idx, blas_idx, ref_val, blas_val);

    reduce(REDUCE_MIN_ABS, A, idx_A.data(), ref_val, ref_idx);
    set(std::abs(data[0]), blas_val);
    blas_idx = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        if (std::abs(data[i]) < blas_val)
        {
            set(std::abs(data[i]), blas_val);
            blas_idx = i;
        }
    }
    passfail("REDUCE_MIN_ABS", ref_idx, blas_idx, ref_val, std::abs(blas_val));
    */

    reduce(REDUCE_NORM_2, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        blas_val += norm2(data[i]);
    }
    blas_val = sqrt(real(blas_val));
    passfail("REDUCE_NORM_2", ref_val, blas_val);

    A = T(1);
    reduce(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 1;
    for (unsigned i = 0;i < A.dimension();i++) blas_val *= A.length(i);
    passfail("COUNT", ref_val, blas_val);
}

template <typename T>
void test(stride_type N_in_bytes, int R)
{
    stride_type N = N_in_bytes/sizeof(T);

    for (int i = 0;i < R;i++) test_tblis<T>(N);

    for (int i = 0;i < R;i++) test_reduce<T>(N);
    for (int i = 0;i < R;i++) test_scale<T>(N);
    for (int i = 0;i < R;i++) test_transpose<T>(N);
    for (int i = 0;i < R;i++) test_dot<T>(N);
    for (int i = 0;i < R;i++) test_replicate<T>(N);
    for (int i = 0;i < R;i++) test_trace<T>(N);
    for (int i = 0;i < R;i++) test_add<T>(N);
    for (int i = 0;i < R;i++) test_outer_prod<T>(N);
    for (int i = 0;i < R;i++) test_weight<T>(N);
    for (int i = 0;i < R;i++) test_contract<T>(N);
    for (int i = 0;i < R;i++) test_mult<T>(N);
}

int main(int argc, char **argv)
{
    stride_type N = 10*1024*1024;
    int R = 10;
    time_t seed = time(NULL);

    struct option opts[] = {{"size", required_argument, NULL, 'n'},
                            {"rep",  required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {0, 0, 0, 0}};

    int arg;
    int index;
    istringstream iss;
    while ((arg = getopt_long(argc, argv, "n:r:s:", opts, &index)) != -1)
    {
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
            case '?':
                ::abort();
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    rand_engine.seed(seed);

    //test<   float>(N, R);
    test<  double>(N, R);
    //test<scomplex>(N, R);
    //test<dcomplex>(N, R);

    return 0;
}
