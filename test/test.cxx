#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <iostream>
#include <random>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <iomanip>

#include "tblis.hpp"

using namespace std;
using namespace stl_ext;
using namespace MArray;
using namespace tblis;
using namespace tblis::impl;
using namespace tblis::util;

namespace tblis
{
namespace util
{
mt19937 engine;
}
}

template <typename T>
void gemm_ref(T alpha, const_matrix_view<T> A,
                       const_matrix_view<T> B,
              T  beta,       matrix_view<T> C)
{
    const T* ptr_A = A.data();
    const T* ptr_B = B.data();
          T* ptr_C = C.data();

    idx_type m_A = A.length(0);
    idx_type m_C = C.length(0);
    idx_type n_B = B.length(1);
    idx_type n_C = C.length(1);
    idx_type k_A = A.length(1);
    idx_type k_B = B.length(0);

    stride_type rs_A = A.stride(0);
    stride_type cs_A = A.stride(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);
    stride_type rs_C = C.stride(0);
    stride_type cs_C = C.stride(1);

    assert(m_A == m_C);
    assert(n_B == n_C);
    assert(k_A == k_B);

    idx_type m = m_A;
    idx_type n = n_B;
    idx_type k = k_A;

    for (idx_type im = 0;im < m;im++)
    {
        for (idx_type in = 0;in < n;in++)
        {
            T tmp = T();

            for (idx_type ik = 0;ik < k;ik++)
            {
                tmp += ptr_A[im*rs_A + ik*cs_A]*ptr_B[ik*rs_B + in*cs_B];
            }

            if (beta == 0.0)
            {
                ptr_C[im*rs_C + in*cs_C] = alpha*tmp;
            }
            else
            {
                ptr_C[im*rs_C + in*cs_C] = alpha*tmp + beta*ptr_C[im*rs_C + in*cs_C];
            }
        }
    }
}

string permutation(string from, string to)
{
    assert(from.size() == to.size());

    string p = from;

    for (size_t i = 0;i < to.size();i++)
    {
        for (size_t j = 0;j < from.size();j++)
        {
            if (from[j] == to[i])
            {
                p[i] = 'a'+j;
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
    bool pass = (sizeof(c) == 4 ? c < 1e-3 : c < 1e-12) && ia == ib;

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

template <typename T> const string& TypeName();

template <> const string& TypeName<float>()
{
    static string name = "float";
    return name;
}

template <> const string& TypeName<double>()
{
    static string name = "double";
    return name;
}

template <> const string& TypeName<scomplex>()
{
    static string name = "scomplex";
    return name;
}

template <> const string& TypeName<dcomplex>()
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
void RandomMatrix(size_t N, idx_type m_min, idx_type n_min, matrix<T>& t)
{
    vector<idx_type> len = RandomProductConstrainedSequence<idx_type>(2, N, {m_min, n_min});

    idx_type m = (m_min > 0 ? m_min : RandomInteger(1, len[0]));
    idx_type n = (n_min > 0 ? n_min : RandomInteger(1, len[1]));

    t.reset({m, n});

    T* data = t.data();
    miterator<2> it(t.lengths(), t.strides());
    while (it.next(data)) *data = RandomUnit<T>();
}

/*
 * Creates a matrix, whose total storage size is between N/4
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/16 and N/4. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void RandomMatrix(size_t N, matrix<T>& t)
{
    RandomMatrix(N, 0, 0, t);
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void RandomTensor(size_t N, unsigned d, vector<idx_type> len_min, tensor<T>& t)
{
    vector<idx_type> len_max = RandomProductConstrainedSequence<idx_type>(d, N, len_min);

    vector<idx_type> len(d);
    for (unsigned i = 0;i < d;i++)
    {
        len[i] = (len_min[i] > 0 ? len_min[i] : RandomInteger(1, len_max[i]));
    }

    t.reset(len);

    T* data = t.data();
    viterator<> it(t.lengths(), t.strides());
    while (it.next(data)) *data = RandomUnit<T>();
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4 and N/2. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void RandomTensor(size_t N, unsigned d, tensor<T>& t)
{
    RandomTensor(N, d, vector<idx_type>(d), t);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
template <typename T>
void RandomTensor(size_t N, tensor<T>& t)
{
    RandomTensor(N, RandomInteger(1,8), t);
}

enum IndexType
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
void RandomTensors(size_t N,
                   unsigned ndim_A_only, unsigned ndim_B_only,
                   unsigned ndim_AB,
                   tensor<T>& A, string& idx_A,
                   tensor<T>& B, string& idx_B)
{
    unsigned ndim_A = ndim_A_only+ndim_AB;
    unsigned ndim_B = ndim_B_only+ndim_AB;

    vector<pair<IndexType,unsigned>> types_A(ndim_A);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_A_only;j++) types_A[i++] = {TYPE_A, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_A[i++] = {TYPE_AB, j};
    }
    random_shuffle(types_A.begin(), types_A.end());

    vector<pair<IndexType,unsigned>> types_B(ndim_B);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_B_only;j++) types_B[i++] = {TYPE_B, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_B[i++] = {TYPE_AB, j};
    }
    random_shuffle(types_B.begin(), types_B.end());

    string idx;
    for (unsigned i = 0;i < ndim_A+ndim_B-ndim_AB;i++) idx.push_back('a'+i);
    random_shuffle(idx.begin(), idx.end());

    unsigned c = 0;
    vector<char> idx_A_only(ndim_A_only);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    vector<char> idx_B_only(ndim_B_only);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    vector<char> idx_AB(ndim_AB);
    for (unsigned i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    idx_A.resize(ndim_A);
    idx_B.resize(ndim_B);

    for (unsigned i = 0;i < ndim_A;i++)
    {
        switch (types_A[i].first)
        {
            case TYPE_A  : idx_A[i] = idx_A_only[types_A[i].second]; break;
            case TYPE_AB : idx_A[i] = idx_AB    [types_A[i].second]; break;
            default: break;
        }
    }

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

    RandomTensor(N, ndim_A, A);

    vector<idx_type> min_B(ndim_B);
    for (unsigned i = 0;i < ndim_B;i++)
    {
        for (unsigned j = 0;j < ndim_A;j++)
        {
            if (idx_B[i] == idx_A[j]) min_B[i] = A.length(j);
        }
    }

    RandomTensor(N, ndim_B, min_B, B);

    if (switch_AB)
    {
        swap(ndim_A, ndim_B);
        swap(idx_A, idx_B);
        swap(A, B);
    }
}

template <typename T>
void RandomTensors(size_t N,
                   unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                   unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                   unsigned ndim_ABC,
                   tensor<T>& A, string& idx_A,
                   tensor<T>& B, string& idx_B,
                   tensor<T>& C, string& idx_C)
{
    unsigned ndim_A = ndim_A_only+ndim_AB+ndim_AC+ndim_ABC;
    unsigned ndim_B = ndim_B_only+ndim_AB+ndim_BC+ndim_ABC;
    unsigned ndim_C = ndim_C_only+ndim_AC+ndim_BC+ndim_ABC;

    vector<pair<IndexType,unsigned>> types_A(ndim_A);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_A_only;j++) types_A[i++] = {TYPE_A, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_A[i++] = {TYPE_AB, j};
        for (unsigned j = 0;j < ndim_AC    ;j++) types_A[i++] = {TYPE_AC, j};
        for (unsigned j = 0;j < ndim_ABC   ;j++) types_A[i++] = {TYPE_ABC, j};
    }
    random_shuffle(types_A.begin(), types_A.end());

    vector<pair<IndexType,unsigned>> types_B(ndim_B);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_B_only;j++) types_B[i++] = {TYPE_B, j};
        for (unsigned j = 0;j < ndim_AB    ;j++) types_B[i++] = {TYPE_AB, j};
        for (unsigned j = 0;j < ndim_BC    ;j++) types_B[i++] = {TYPE_BC, j};
        for (unsigned j = 0;j < ndim_ABC   ;j++) types_B[i++] = {TYPE_ABC, j};
    }
    random_shuffle(types_B.begin(), types_B.end());

    vector<pair<IndexType,unsigned>> types_C(ndim_C);
    {
        unsigned i = 0;
        for (unsigned j = 0;j < ndim_C_only;j++) types_C[i++] = {TYPE_C, j};
        for (unsigned j = 0;j < ndim_AC    ;j++) types_C[i++] = {TYPE_AC, j};
        for (unsigned j = 0;j < ndim_BC    ;j++) types_C[i++] = {TYPE_BC, j};
        for (unsigned j = 0;j < ndim_ABC   ;j++) types_C[i++] = {TYPE_ABC, j};
    }
    random_shuffle(types_C.begin(), types_C.end());

    string idx;
    for (unsigned i = 0;i < ndim_A_only+ndim_B_only+ndim_C_only+
                            ndim_AB+ndim_AC+ndim_BC+ndim_ABC;i++) idx.push_back('a'+i);
    random_shuffle(idx.begin(), idx.end());

    unsigned c = 0;
    vector<char> idx_A_only(ndim_A_only);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    vector<char> idx_B_only(ndim_B_only);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    vector<char> idx_C_only(ndim_C_only);
    for (unsigned i = 0;i < ndim_C_only;i++) idx_C_only[i] = idx[c++];

    vector<char> idx_AB(ndim_AB);
    for (unsigned i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    vector<char> idx_AC(ndim_AC);
    for (unsigned i = 0;i < ndim_AC;i++) idx_AC[i] = idx[c++];

    vector<char> idx_BC(ndim_BC);
    for (unsigned i = 0;i < ndim_BC;i++) idx_BC[i] = idx[c++];

    vector<char> idx_ABC(ndim_ABC);
    for (unsigned i = 0;i < ndim_ABC;i++) idx_ABC[i] = idx[c++];

    idx_A.resize(ndim_A);
    idx_B.resize(ndim_B);
    idx_C.resize(ndim_C);

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
        RandomTensor(N, ndim_A, A);

        vector<idx_type> min_B(ndim_B);
        for (unsigned i = 0;i < ndim_B;i++)
        {
            for (unsigned j = 0;j < ndim_A;j++)
            {
                if (idx_B[i] == idx_A[j]) min_B[i] = A.length(j);
            }
        }

        RandomTensor(N, ndim_B, min_B, B);

        stride_type siz = 1;
        vector<idx_type> min_C(ndim_C);
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

        RandomTensor(N, ndim_C, min_C, C);

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
 * Creates a random tensor summation operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomSum(size_t N, tensor<T>& A, string& idx_A,
                         tensor<T>& B, string& idx_B)
{
    unsigned ndim_A = RandomInteger(1,8);
    unsigned ndim_B = RandomInteger(1,8);

    unsigned ndim_AB = RandomInteger(0,min(ndim_A,ndim_B));
    unsigned ndim_A_only = ndim_A-ndim_AB;
    unsigned ndim_B_only = ndim_B-ndim_AB;

    RandomTensors(N,
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
void RandomTrace(size_t N, tensor<T>& A, string& idx_A,
                           tensor<T>& B, string& idx_B)
{
    unsigned ndim_A = RandomInteger(1,8);
    unsigned ndim_B = RandomInteger(1,8);

    if (ndim_A < ndim_B) swap(ndim_A, ndim_B);

    RandomTensors(N,
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
void RandomReplicate(size_t N, tensor<T>& A, string& idx_A,
                               tensor<T>& B, string& idx_B)
{
    unsigned ndim_A = RandomInteger(1,8);
    unsigned ndim_B = RandomInteger(1,8);

    if (ndim_B < ndim_A) swap(ndim_A, ndim_B);

    RandomTensors(N,
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
void RandomTranspose(size_t N, tensor<T>& A, string& idx_A,
                               tensor<T>& B, string& idx_B)
{
    unsigned ndim_A = RandomInteger(1,8);

    RandomTensors(N,
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
void RandomDot(size_t N, tensor<T>& A, string& idx_A,
                         tensor<T>& B, string& idx_B)
{
    unsigned ndim_A = RandomInteger(1,8);

    RandomTensors(N,
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
void RandomMult(size_t N, tensor<T>& A, string& idx_A,
                          tensor<T>& B, string& idx_B,
                          tensor<T>& C, string& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_A_only, ndim_B_only, ndim_C_only;
    int ndim_AB, ndim_AC, ndim_BC;
    int ndim_ABC;
    do
    {
        ndim_A = RandomInteger(1,8);
        ndim_B = RandomInteger(1,8);
        ndim_C = RandomInteger(1,8);
        ndim_A_only = RandomInteger(    ndim_A);
        ndim_B_only = RandomInteger(    ndim_B);
        ndim_C_only = RandomInteger(    ndim_C);
        ndim_ABC    = RandomInteger(min(ndim_A,
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

    RandomTensors(N,
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
void RandomGEMM(size_t N, matrix<T>& A,
                          matrix<T>& B,
                          matrix<T>& C)
{
    idx_type m = RandomInteger(1, (idx_type)sqrt(N));
    idx_type n = RandomInteger(1, (idx_type)sqrt(N));
    idx_type k = RandomInteger(1, (idx_type)sqrt(N));

    //m += (MR<T>::value-1)-(m-1)%MR<T>::value;
    //n += (NR<T>::value-1)-(n-1)%NR<T>::value;
    //k += (KR<T>::value-1)-(k-1)%KR<T>::value;

    //m = 46;
    //n = 334;
    //k = 28;

    //engine.seed(0);

    RandomMatrix(N, m, k, A);
    RandomMatrix(N, k, n, B);
    RandomMatrix(N, m, n, C);

    //printf("%.15f %.15f\n", (double)real(tblis_normfm(A)), (double)real(tblis_normfm(B)));
}

/*
 * Creates a random matrix times vector operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomGEMV(size_t N, matrix<T>& A,
                          matrix<T>& B,
                          matrix<T>& C)
{
    idx_type m = RandomInteger(1, (idx_type)sqrt(N));
    idx_type k = RandomInteger(1, (idx_type)sqrt(N));

    RandomMatrix(N, m, k, A);
    RandomMatrix(N, k, 1, B);
    RandomMatrix(N, m, 1, C);
}

/*
 * Creates a random matrix outer product operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomGER(size_t N, matrix<T>& A,
                         matrix<T>& B,
                         matrix<T>& C)
{
    idx_type m = RandomInteger(1, (idx_type)sqrt(N));
    idx_type n = RandomInteger(1, (idx_type)sqrt(N));

    RandomMatrix(N, m, 1, A);
    RandomMatrix(N, 1, n, B);
    RandomMatrix(N, m, n, C);
}

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomContract(size_t N, tensor<T>& A, string& idx_A,
                              tensor<T>& B, string& idx_B,
                              tensor<T>& C, string& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AB, ndim_AC, ndim_BC;
    do
    {
        ndim_A = RandomInteger(1,8);
        ndim_B = RandomInteger(1,8);
        ndim_C = RandomInteger(1,8);
        ndim_AB = (ndim_A+ndim_B-ndim_C)/2;
        ndim_AC = ndim_A-ndim_AB;
        ndim_BC = ndim_B-ndim_AB;
    }
    while (ndim_AB < 0 ||
           ndim_AC < 0 ||
           ndim_BC < 0 ||
           (ndim_A+ndim_B+ndim_C)%2 != 0);

    RandomTensors(N,
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
void RandomWeight(size_t N, tensor<T>& A, string& idx_A,
                            tensor<T>& B, string& idx_B,
                            tensor<T>& C, string& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AC, ndim_BC;
    int ndim_ABC;
    do
    {
        ndim_A = RandomInteger(1,8);
        ndim_B = RandomInteger(1,8);
        ndim_C = RandomInteger(1,8);
        ndim_ABC = ndim_A+ndim_B-ndim_C;
        ndim_AC = ndim_A-ndim_ABC;
        ndim_BC = ndim_B-ndim_ABC;
    }
    while (ndim_AC  < 0 ||
           ndim_BC  < 0 ||
           ndim_ABC < 0);

    RandomTensors(N,
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
void RandomOuterProd(size_t N, tensor<T>& A, string& idx_A,
                               tensor<T>& B, string& idx_B,
                               tensor<T>& C, string& idx_C)
{
    unsigned ndim_A, ndim_B, ndim_C;
    do
    {
        ndim_A = RandomInteger(1,8);
        ndim_B = RandomInteger(1,8);
        ndim_C = ndim_A+ndim_B;
    }
    while (ndim_C > 8);

    RandomTensors(N,
                  0, 0, 0,
                  0, ndim_A, ndim_B,
                  0,
                  A, idx_A,
                  B, idx_B,
                  C, idx_C);
}

template <typename T>
void TestTBLIS(size_t N)
{
    matrix<T> A, B, C, D;

    for (int pass = 0;pass < 3;pass++)
    {
        switch (pass)
        {
            case 0: RandomGEMM(N, A, B, C); break;
            case 1: RandomGEMV(N, A, B, C); break;
            case 2: RandomGER (N, A, B, C); break;
        }

        D.reset(C);

        T ref_val, calc_val;
        T scale = 10.0*RandomUnit<T>();

        //repeat:

        cout << endl;
        cout << "Testing TBLIS/" << (pass == 0 ? "GEMM" :
                                     pass == 1 ? "GEMV" :
                                                 "GER") << " (" << TypeName<T>() << "):" << endl;

        cout << endl;
        cout << "m, n, k    = " << C.length(0) << ", " << C.length(1) << ", " << A.length(1) << endl;
        cout << "rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1) << endl;
        cout << "rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1) << endl;
        cout << "rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1) << endl;
        cout << endl;

        //printf("A: %f\n", pow((double)real(tblis_normfm(A)),2));
        //printf("B: %f\n", pow((double)real(tblis_normfm(B)),2));
        //printf("\n");

        D = C;
        gemm_ref(scale, A, B, scale, D);
        tblis_normfm(D, ref_val);

        D = C;
        tblis_gemm(scale, A, B, scale, D);
        tblis_normfm(D, calc_val);

        passfail("REF", ref_val, calc_val);
        //exit(0);

        //goto repeat;
    }
}

template <> void TestTBLIS<scomplex>(size_t N) {}

template <> void TestTBLIS<dcomplex>(size_t N) {}

template <typename T>
void TestMult(size_t N)
{
    tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    cout << endl;
    cout << "Testing mult (" << TypeName<T>() << "):" << endl;

    RandomContract(N, A, idx_A, B, idx_B, C, idx_C);

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

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D.reset(C);
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("CONTRACT", ref_val, calc_val);

    RandomWeight(N, A, idx_A, B, idx_B, C, idx_C);

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

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_weight(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D.reset(C);
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("WEIGHT", ref_val, calc_val);

    RandomOuterProd(N, A, idx_A, B, idx_B, C, idx_C);

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

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_outer_prod(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D.reset(C);
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("OUTER_PROD", ref_val, calc_val);

    RandomMult(N, A, idx_A, B, idx_B, C, idx_C);

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

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D.reset(C);
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestContract(size_t N)
{
    tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    RandomContract(N, A, idx_A, B, idx_B, C, idx_C);

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    //if (idx_C != "ghdfeba") return;

    cout << endl;
    cout << "Testing contract (" << TypeName<T>() << "):" << endl;
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

    impl_type = REFERENCE;
    D.reset(C);
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    impl_type = REFERENCE;
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);

    impl_type = BLIS_BASED;
    D.reset(C);
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    impl_type = REFERENCE;
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLIS", ref_val, calc_val);

    //if (idx_C == "ghdfeba") exit(0);
}

template <typename T>
void TestWeight(size_t N)
{
    tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    RandomWeight(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing weight (" << TypeName<T>() << "):" << endl;
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

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_weight(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D.reset(C);
    tensor_weight(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestOuterProd(size_t N)
{
    tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    RandomOuterProd(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing outer prod (" << TypeName<T>() << "):" << endl;
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

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    impl_type = BLAS_BASED;
    D.reset(C);
    tensor_outer_prod(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D.reset(C);
    tensor_outer_prod(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestSum(size_t N)
{
    tensor<T> A, B, C;
    string idx_A, idx_B;

    T ref_val, calc_val, scale, sum_b;
    scale = 10.0*RandomUnit<T>();

    cout << endl;
    cout << "Testing sum (" << TypeName<T>() << "):" << endl;

    RandomTranspose(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    C.reset(B);
    tensor_transpose(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C.reset(B);
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("TRANSPOSE", ref_val, calc_val);

    RandomTrace(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    C.reset(B);
    tensor_trace(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C.reset(B);
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("TRACE", ref_val, calc_val);

    RandomReplicate(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    C.reset(B);
    tensor_replicate(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C.reset(B);
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("REPLICATE", ref_val, calc_val);

    RandomSum(N, A, idx_A, B, idx_B);

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

    impl_type = REFERENCE;
    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);
    tensor_reduce(REDUCE_SUM, B, idx_B, sum_b);
    tensor_sum(scale, A, idx_A, scale, B, idx_B);
    tensor_reduce(REDUCE_SUM, B, idx_B, calc_val);
    passfail("SUM", scale*(sz*ref_val+sum_b), calc_val);

    impl_type = BLAS_BASED;
    C.reset(B);
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C.reset(B);
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestTrace(size_t N)
{
    tensor<T> A, B;
    string idx_A, idx_B;

    RandomTrace(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing trace (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, sum_b, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);
    tensor_reduce(REDUCE_SUM, B, idx_B, sum_b);
    tensor_trace(scale, A, idx_A, scale, B, idx_B);
    tensor_reduce(REDUCE_SUM, B, idx_B, calc_val);
    passfail("SUM", scale*(ref_val+sum_b), calc_val);
}

template <typename T>
void TestReplicate(size_t N)
{
    tensor<T> A, B;
    string idx_A, idx_B;

    RandomReplicate(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing replicate (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, sum_b, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

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

    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);
    tensor_reduce(REDUCE_SUM, B, idx_B, sum_b);
    tensor_replicate(scale, A, idx_A, scale, B, idx_B);
    tensor_reduce(REDUCE_SUM, B, idx_B, calc_val);
    passfail("SUM", scale*(sz*ref_val+sum_b), calc_val);

    tensor_reduce(REDUCE_NORM_1, A, idx_A, ref_val);
    tensor_replicate(scale, A, idx_A, T(0.0), B, idx_B);
    tensor_reduce(REDUCE_NORM_1, B, idx_B, calc_val);
    passfail("NRM1", sz*T(std::abs(scale))*ref_val, calc_val);
}

template <typename T>
void TestDot(size_t N)
{
    tensor<T> A, B;
    string idx_A, idx_B;

    RandomDot(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing dot (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, calc_val;

    tensor_transpose(T(1.0), A, idx_A, T(0.0), B, idx_B);
    T* data = B.data();
    viterator<> it(B.lengths(), B.strides());
    while (it.next(data)) *data = stl_ext::conj(*data);
    tensor_reduce(REDUCE_NORM_2, A, idx_A, ref_val);
    tensor_dot(A, idx_A, B, idx_B, calc_val);
    passfail("NRM2", ref_val*ref_val, calc_val);

    B = T(1);
    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);
    tensor_dot(A, idx_A, B, idx_B, calc_val);
    passfail("UNIT", ref_val, calc_val);

    B = T(0);
    tensor_dot(A, idx_A, B, idx_B, calc_val);
    passfail("ZERO", T(0.0), calc_val);
}

template <typename T>
void TestTranspose(size_t N)
{
    tensor<T> A, B, C;
    string idx_A, idx_B;

    RandomTranspose(N, A, idx_A, B, idx_B);

    unsigned ndim = A.dimension();
    string perm = permutation(idx_A, idx_B);

    cout << endl;
    cout << "Testing transpose (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << "perm   = " << perm << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, calc_val, scale;

    scale = 10.0*RandomUnit<T>();

    C.reset(A);
    string idx_C = idx_A;
    tensor_reduce(REDUCE_NORM_2, A, idx_A, ref_val);
    tensor_transpose(T(1), A, idx_A, T(0), B, idx_B);
    tensor_transpose(scale, B, idx_B, scale, C, idx_C);
    tensor_reduce(REDUCE_NORM_2, C, idx_C, calc_val);
    passfail("INVERSE", T(2.0*std::abs(scale))*ref_val, calc_val);

    B.reset(A);
    idx_B = idx_A;
    vector<idx_type> len_C(ndim);
    do
    {
        for (unsigned i = 0;i < ndim;i++)
        {
            unsigned j; for (j = 0;j < ndim && idx_A[j] != perm[i];j++) continue;
            idx_C[i] = idx_B[j];
            len_C[i] = B.length(j);
        }
        C.reset(len_C);
        tensor_transpose(T(1), B, idx_B, T(0), C, idx_C);
        B.reset(C);
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    tensor_reduce(REDUCE_NORM_2, C, idx_C, calc_val);
    passfail("CYCLE", ref_val, calc_val);
}

template <typename T>
void TestScale(size_t N)
{
    tensor<T> A;
    string idx_A = "abcdefgh";

    RandomTensor(N, A);
    idx_A.resize(A.dimension());

    cout << endl;
    cout << "Testing scale (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, calc_val, scale;

    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);

    scale = 10.0*RandomUnit<T>();
    tensor_scale(scale, A, idx_A);
    tensor_reduce(REDUCE_SUM, A, idx_A, calc_val);
    passfail("RANDOM", ref_val*scale, calc_val);

    tensor_scale(T(1.0), A, idx_A);
    tensor_reduce(REDUCE_SUM, A, idx_A, calc_val);
    passfail("UNIT", ref_val*scale, calc_val);

    tensor_scale(T(0.0), A, idx_A);
    tensor_reduce(REDUCE_SUM, A, idx_A, calc_val);
    passfail("ZERO", T(0.0), calc_val);
}

template <typename T>
void TestReduce(size_t N)
{
    tensor<T> A;
    string idx_A = "abcdefgh";

    RandomTensor(N, A);
    idx_A.resize(A.dimension());
    size_t NA = A.stride(A.dimension()-1)*A.length(A.dimension()-1);

    cout << endl;
    cout << "Testing reduction (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val, ref_idx);
    blas_val = 0;
    for (size_t i = 0;i < NA;i++)
    {
        blas_val += data[i];
    }
    passfail("REDUCE_SUM", ref_val, blas_val);

    tensor_reduce(REDUCE_SUM_ABS, A, idx_A, ref_val, ref_idx);
    blas_val = 0;
    for (size_t i = 0;i < NA;i++)
    {
        blas_val += std::abs(data[i]);
    }
    passfail("REDUCE_SUM_ABS", ref_val, blas_val);

    tensor_reduce(REDUCE_MAX, A, idx_A, ref_val, ref_idx);
    blas_val = data[0];
    blas_idx = 0;
    for (size_t i = 0;i < NA;i++)
    {
        if (data[i] > blas_val)
        {
            blas_val = data[i];
            blas_idx = i;
        }
    }
    passfail("REDUCE_MAX", ref_idx, blas_idx, ref_val, blas_val);

    tensor_reduce(REDUCE_MAX_ABS, A, idx_A, ref_val, ref_idx);
    blas_val = std::abs(data[0]);
    blas_idx = 0;
    for (size_t i = 0;i < NA;i++)
    {
        if (std::abs(data[i]) > blas_val)
        {
            blas_val = std::abs(data[i]);
            blas_idx = i;
        }
    }
    passfail("REDUCE_MAX_ABS", ref_idx, blas_idx, ref_val, blas_val);

    /*
    tensor_reduce(REDUCE_MIN, A, idx_A, ref_val, ref_idx);
    set(data[0], blas_val);
    blas_idx = 0;
    for (size_t i = 0;i < NA;i++)
    {
        if (data[i] < blas_val)
        {
            set(data[i], blas_val);
            blas_idx = i;
        }
    }
    passfail("REDUCE_MIN", ref_idx, blas_idx, ref_val, blas_val);

    tensor_reduce(REDUCE_MIN_ABS, A, idx_A, ref_val, ref_idx);
    set(std::abs(data[0]), blas_val);
    blas_idx = 0;
    for (size_t i = 0;i < NA;i++)
    {
        if (std::abs(data[i]) < blas_val)
        {
            set(std::abs(data[i]), blas_val);
            blas_idx = i;
        }
    }
    passfail("REDUCE_MIN_ABS", ref_idx, blas_idx, ref_val, std::abs(blas_val));
    */

    tensor_reduce(REDUCE_NORM_2, A, idx_A, ref_val, ref_idx);
    blas_val = 0;
    for (size_t i = 0;i < NA;i++)
    {
        blas_val += norm2(data[i]);
    }
    blas_val = sqrt(real(blas_val));
    passfail("REDUCE_NORM_2", ref_val, blas_val);

    A = T(1);
    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val, ref_idx);
    blas_val = 1;
    for (int i = 0;i < A.dimension();i++) blas_val *= A.length(i);
    passfail("COUNT", ref_val, blas_val);
}

template <typename T>
void Test(size_t N_in_bytes, int R)
{
    size_t N = N_in_bytes/sizeof(T);

    for (int i = 0;i < R;i++) TestTBLIS<T>(N);

    for (int i = 0;i < R;i++) TestReduce<T>(N);
    for (int i = 0;i < R;i++) TestScale<T>(N);
    for (int i = 0;i < R;i++) TestTranspose<T>(N);
    for (int i = 0;i < R;i++) TestDot<T>(N);
    for (int i = 0;i < R;i++) TestReplicate<T>(N);
    for (int i = 0;i < R;i++) TestTrace<T>(N);
    for (int i = 0;i < R;i++) TestSum<T>(N);
    //for (int i = 0;i < R;i++) TestOuterProd<T>(N);
    //for (int i = 0;i < R;i++) TestWeight<T>(N);
    for (int i = 0;i < R;i++) TestContract<T>(N);
    //for (int i = 0;i < R;i++) TestMult<T>(N);
}

int main(int argc, char **argv)
{
    size_t N = 1024*1024;
    int R = 10;
    time_t seed = time(NULL);

    tblis_init();

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
                break;
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    engine.seed(seed);

    //Test<   float>(N, R);
    Test<  double>(N, R);
    //Test<scomplex>(N, R);
    //Test<dcomplex>(N, R);

    tblis_finalize();

    return 0;
}
