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
#include "util/util.hpp"

using namespace std;
using namespace blis;
using namespace tblis;
using namespace tblis::impl;
using namespace tblis::util;
using namespace tblis::blis_like;

namespace tblis
{
namespace util
{
mt19937 engine;
}
}

template <typename T>
void gemm_ref(T alpha, obj_t* A, obj_t* B, T beta, obj_t* C)
{
    T* ptr_A = (T*)bli_obj_buffer(*A);
    T* ptr_B = (T*)bli_obj_buffer(*B);
    T* ptr_C = (T*)bli_obj_buffer(*C);

    dim_t m_A = bli_obj_length(*A);
    dim_t m_C = bli_obj_length(*C);
    dim_t n_B = bli_obj_width(*B);
    dim_t n_C = bli_obj_width(*C);
    dim_t k_A = bli_obj_width(*A);
    dim_t k_B = bli_obj_length(*B);

    inc_t rs_A = bli_obj_row_stride(*A);
    inc_t cs_A = bli_obj_col_stride(*A);
    inc_t rs_B = bli_obj_row_stride(*B);
    inc_t cs_B = bli_obj_col_stride(*B);
    inc_t rs_C = bli_obj_row_stride(*C);
    inc_t cs_C = bli_obj_col_stride(*C);

    if (bli_obj_has_trans(*A))
    {
        swap(m_A, k_A);
        swap(rs_A, cs_A);
    }

    if (bli_obj_has_trans(*B))
    {
        swap(k_B, n_B);
        swap(rs_B, cs_B);
    }

    if (bli_obj_has_trans(*C))
    {
        swap(m_C, n_C);
        swap(rs_C, cs_C);
    }

    ASSERT(m_A == m_C);
    ASSERT(n_B == n_C);
    ASSERT(k_A == k_B);

    dim_t m = m_A;
    dim_t n = n_B;
    dim_t k = k_A;

    for (dim_t im = 0;im < m;im++)
    {
        for (dim_t in = 0;in < n;in++)
        {
            T tmp = T();

            for (dim_t ik = 0;ik < k;ik++)
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

    for (gint_t i = 0;i < to.size();i++)
    {
        for (gint_t j = 0;j < from.size();j++)
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
void passfail(const string& label, inc_t ia, inc_t ib, T a, U b)
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
        abort();
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

template <> const string& TypeName<sComplex>()
{
    static string name = "sComplex";
    return name;
}

template <> const string& TypeName<dComplex>()
{
    static string name = "dComplex";
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
void RandomMatrix(siz_t N, dim_t m_min, dim_t n_min, Matrix<T>& t)
{
    vector<inc_t> stride = RandomProductConstrainedSequence<inc_t>(3, N, {1, m_min, n_min});

    dim_t m = (m_min > 0 ? m_min : RandomInteger(1, stride[1]));
    dim_t n = (n_min > 0 ? n_min : RandomInteger(1, stride[2]));

    dim_t rs = stride[0];
    dim_t cs = stride[1]*rs;

    /*
    if (rs == cs)
    {
        ASSERT(m == 1 || n == 1);
        (m == 1 ? rs : cs) = 1;
    }
    */

    siz_t size = 1+(m-1)*rs+(n-1)*cs;

    t.reset(m, n, rs, cs);

    T* data = t.data();
    fill(data, data+size, T());

    Iterator it({m,n}, {rs,cs});
    while (it.nextIteration(data)) *data = RandomUnit<T>();
}

/*
 * Creates a matrix, whose total storage size is between N/4
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/16 and N/4. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void RandomMatrix(siz_t N, Matrix<T>& t)
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
void RandomTensor(siz_t N, gint_t d, vector<dim_t> len_min, Tensor<T>& t)
{
    len_min.insert(len_min.begin(), 1);
    vector<inc_t> stride = RandomProductConstrainedSequence<inc_t>(d+1, N, len_min);

    vector<dim_t> len(d);
    for (gint_t i = 0;i < d;i++)
    {
        if (len_min[i+1] > 0)
        {
            len[i] = len_min[i+1];
        }
        else
        {
            len[i] = RandomInteger(1, stride[i+1]);
        }
    }

    stride.resize(d);

    siz_t size = 1+(len[0]-1)*stride[0];
    for (gint_t i = 1;i < d;i++)
    {
        stride[i] *= stride[i-1];
        size += (len[i]-1)*stride[i];
    }

    t.reset(d, len, stride);
    assert(size == t.getDataSize());

    T* data = t;
    fill(data, data+size, T());

    Iterator it(len, stride);
    while (it.nextIteration(data)) *data = RandomUnit<T>();
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4 and N/2. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void RandomTensor(siz_t N, gint_t d, Tensor<T>& t)
{
    RandomTensor(N, d, vector<dim_t>(d, 0), t);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
template <typename T>
void RandomTensor(siz_t N, Tensor<T>& t)
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
void RandomTensors(siz_t N,
                   gint_t ndim_A_only, gint_t ndim_B_only,
                   gint_t ndim_AB,
                   Tensor<T>& A, string& idx_A,
                   Tensor<T>& B, string& idx_B)
{
    gint_t ndim_A = ndim_A_only+ndim_AB;
    gint_t ndim_B = ndim_B_only+ndim_AB;

    vector<pair<IndexType,gint_t>> types_A(ndim_A);
    {
        gint_t i = 0;
        for (gint_t j = 0;j < ndim_A_only;j++) types_A[i++] = pair<IndexType,gint_t>(TYPE_A, j);
        for (gint_t j = 0;j < ndim_AB    ;j++) types_A[i++] = pair<IndexType,gint_t>(TYPE_AB, j);
    }
    random_shuffle(types_A.begin(), types_A.end());

    vector<pair<IndexType,gint_t>> types_B(ndim_B);
    {
        gint_t i = 0;
        for (gint_t j = 0;j < ndim_B_only;j++) types_B[i++] = pair<IndexType,gint_t>(TYPE_B, j);
        for (gint_t j = 0;j < ndim_AB    ;j++) types_B[i++] = pair<IndexType,gint_t>(TYPE_AB, j);
    }
    random_shuffle(types_B.begin(), types_B.end());

    string idx;
    for (gint_t i = 0;i < ndim_A+ndim_B-ndim_AB;i++) idx.push_back('a'+i);
    random_shuffle(idx.begin(), idx.end());

    gint_t c = 0;
    vector<char> idx_A_only(ndim_A_only);
    for (gint_t i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    vector<char> idx_B_only(ndim_B_only);
    for (gint_t i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    vector<char> idx_AB(ndim_AB);
    for (gint_t i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    idx_A.resize(ndim_A);
    idx_B.resize(ndim_B);

    for (gint_t i = 0;i < ndim_A;i++)
    {
        switch (types_A[i].first)
        {
            case TYPE_A  : idx_A[i] = idx_A_only[types_A[i].second]; break;
            case TYPE_AB : idx_A[i] = idx_AB    [types_A[i].second]; break;
            default: break;
        }
    }

    for (gint_t i = 0;i < ndim_B;i++)
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

    vector<inc_t> min_B(ndim_B);
    for (gint_t i = 0;i < ndim_B;i++)
    {
        for (gint_t j = 0;j < ndim_A;j++)
        {
            if (idx_B[i] == idx_A[j]) min_B[i] = A.getLength(j);
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
void RandomTensors(siz_t N,
                   gint_t ndim_A_only, gint_t ndim_B_only, gint_t ndim_C_only,
                   gint_t ndim_AB, gint_t ndim_AC, gint_t ndim_BC,
                   gint_t ndim_ABC,
                   Tensor<T>& A, string& idx_A,
                   Tensor<T>& B, string& idx_B,
                   Tensor<T>& C, string& idx_C)
{
    gint_t ndim_A = ndim_A_only+ndim_AB+ndim_AC+ndim_ABC;
    gint_t ndim_B = ndim_B_only+ndim_AB+ndim_BC+ndim_ABC;
    gint_t ndim_C = ndim_C_only+ndim_AC+ndim_BC+ndim_ABC;

    vector<pair<IndexType,gint_t>> types_A(ndim_A);
    {
        gint_t i = 0;
        for (gint_t j = 0;j < ndim_A_only;j++) types_A[i++] = pair<IndexType,gint_t>(TYPE_A, j);
        for (gint_t j = 0;j < ndim_AB    ;j++) types_A[i++] = pair<IndexType,gint_t>(TYPE_AB, j);
        for (gint_t j = 0;j < ndim_AC    ;j++) types_A[i++] = pair<IndexType,gint_t>(TYPE_AC, j);
        for (gint_t j = 0;j < ndim_ABC   ;j++) types_A[i++] = pair<IndexType,gint_t>(TYPE_ABC, j);
    }
    random_shuffle(types_A.begin(), types_A.end());

    vector<pair<IndexType,gint_t>> types_B(ndim_B);
    {
        gint_t i = 0;
        for (gint_t j = 0;j < ndim_B_only;j++) types_B[i++] = pair<IndexType,gint_t>(TYPE_B, j);
        for (gint_t j = 0;j < ndim_AB    ;j++) types_B[i++] = pair<IndexType,gint_t>(TYPE_AB, j);
        for (gint_t j = 0;j < ndim_BC    ;j++) types_B[i++] = pair<IndexType,gint_t>(TYPE_BC, j);
        for (gint_t j = 0;j < ndim_ABC   ;j++) types_B[i++] = pair<IndexType,gint_t>(TYPE_ABC, j);
    }
    random_shuffle(types_B.begin(), types_B.end());

    vector<pair<IndexType,gint_t>> types_C(ndim_C);
    {
        gint_t i = 0;
        for (gint_t j = 0;j < ndim_C_only;j++) types_C[i++] = pair<IndexType,gint_t>(TYPE_C, j);
        for (gint_t j = 0;j < ndim_AC    ;j++) types_C[i++] = pair<IndexType,gint_t>(TYPE_AC, j);
        for (gint_t j = 0;j < ndim_BC    ;j++) types_C[i++] = pair<IndexType,gint_t>(TYPE_BC, j);
        for (gint_t j = 0;j < ndim_ABC   ;j++) types_C[i++] = pair<IndexType,gint_t>(TYPE_ABC, j);
    }
    random_shuffle(types_C.begin(), types_C.end());

    string idx;
    for (gint_t i = 0;i < ndim_A_only+ndim_B_only+ndim_C_only+
                          ndim_AB+ndim_AC+ndim_BC+ndim_ABC;i++) idx.push_back('a'+i);
    random_shuffle(idx.begin(), idx.end());

    gint_t c = 0;
    vector<char> idx_A_only(ndim_A_only);
    for (gint_t i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    vector<char> idx_B_only(ndim_B_only);
    for (gint_t i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    vector<char> idx_C_only(ndim_C_only);
    for (gint_t i = 0;i < ndim_C_only;i++) idx_C_only[i] = idx[c++];

    vector<char> idx_AB(ndim_AB);
    for (gint_t i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    vector<char> idx_AC(ndim_AC);
    for (gint_t i = 0;i < ndim_AC;i++) idx_AC[i] = idx[c++];

    vector<char> idx_BC(ndim_BC);
    for (gint_t i = 0;i < ndim_BC;i++) idx_BC[i] = idx[c++];

    vector<char> idx_ABC(ndim_ABC);
    for (gint_t i = 0;i < ndim_ABC;i++) idx_ABC[i] = idx[c++];

    idx_A.resize(ndim_A);
    idx_B.resize(ndim_B);
    idx_C.resize(ndim_C);

    for (gint_t i = 0;i < ndim_A;i++)
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

    for (gint_t i = 0;i < ndim_B;i++)
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

    for (gint_t i = 0;i < ndim_C;i++)
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

        vector<inc_t> min_B(ndim_B);
        for (gint_t i = 0;i < ndim_B;i++)
        {
            for (gint_t j = 0;j < ndim_A;j++)
            {
                if (idx_B[i] == idx_A[j]) min_B[i] = A.getLength(j);
            }
        }

        RandomTensor(N, ndim_B, min_B, B);

        inc_t siz = 1;
        vector<inc_t> min_C(ndim_C);
        for (gint_t i = 0;i < ndim_C;i++)
        {
            for (gint_t j = 0;j < ndim_A;j++)
            {
                if (idx_C[i] == idx_A[j])
                {
                    min_C[i] = A.getLength(j);
                    siz *= min_C[i];
                }
            }
            for (gint_t j = 0;j < ndim_B;j++)
            {
                if (idx_C[i] == idx_B[j])
                {
                    min_C[i] = B.getLength(j);
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
void RandomSum(siz_t N, Tensor<T>& A, string& idx_A,
                        Tensor<T>& B, string& idx_B)
{
    gint_t ndim_A = RandomInteger(1,8);
    gint_t ndim_B = RandomInteger(1,8);

    gint_t ndim_AB = RandomInteger(0,min(ndim_A,ndim_B));
    gint_t ndim_A_only = ndim_A-ndim_AB;
    gint_t ndim_B_only = ndim_B-ndim_AB;

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
void RandomTrace(siz_t N, Tensor<T>& A, string& idx_A,
                          Tensor<T>& B, string& idx_B)
{
    gint_t ndim_A = RandomInteger(1,8);
    gint_t ndim_B = RandomInteger(1,8);

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
void RandomReplicate(siz_t N, Tensor<T>& A, string& idx_A,
                              Tensor<T>& B, string& idx_B)
{
    gint_t ndim_A = RandomInteger(1,8);
    gint_t ndim_B = RandomInteger(1,8);

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
void RandomTranspose(siz_t N, Tensor<T>& A, string& idx_A,
                              Tensor<T>& B, string& idx_B)
{
    gint_t ndim_A = RandomInteger(1,8);

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
void RandomDot(siz_t N, Tensor<T>& A, string& idx_A,
                        Tensor<T>& B, string& idx_B)
{
    gint_t ndim_A = RandomInteger(1,8);

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
void RandomMult(siz_t N, Tensor<T>& A, string& idx_A,
                         Tensor<T>& B, string& idx_B,
                         Tensor<T>& C, string& idx_C)
{
    gint_t ndim_A, ndim_B, ndim_C;
    gint_t ndim_A_only, ndim_B_only, ndim_C_only;
    gint_t ndim_AB, ndim_AC, ndim_BC;
    gint_t ndim_ABC;
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
void RandomGEMM(siz_t N, Matrix<T>& A,
                         Matrix<T>& B,
                         Matrix<T>& C)
{
    dim_t m = RandomInteger(1, (dim_t)sqrt(N));
    dim_t n = RandomInteger(1, (dim_t)sqrt(N));
    dim_t k = RandomInteger(1, (dim_t)sqrt(N));

    //m += (MR<T>::value-1)-(m-1)%MR<T>::value;
    //n += (NR<T>::value-1)-(n-1)%NR<T>::value;
    //k += (KR<T>::value-1)-(k-1)%KR<T>::value;

    RandomMatrix(N, m, k, A);
    RandomMatrix(N, k, n, B);
    RandomMatrix(N, m, n, C);

    switch (RandomInteger(5))
    {
        case 0: // ABC -> ABC
            break;
        case 1: // ABC -> ACB
            swap(B, C);
            A.transpose();
            break;
        case 2: // ABC -> BAC
            swap(A, B);
            A.transpose();
            B.transpose();
            C.transpose();
            break;
        case 3: // ABC -> BCA
            swap(A, B);
            swap(B, C);
            B.transpose();
            C.transpose();
            break;
        case 4: // ABC -> CAB
            swap(A, B);
            swap(A, C);
            A.transpose();
            C.transpose();
            break;
        case 5: // ABC -> CBA
            swap(A, C);
            B.transpose();
            break;
    }
}

/*
 * Creates a random matrix times vector operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomGEMV(siz_t N, Matrix<T>& A,
                         Matrix<T>& B,
                         Matrix<T>& C)
{
    dim_t m = RandomInteger(1, (dim_t)sqrt(N));
    dim_t k = RandomInteger(1, (dim_t)sqrt(N));

    RandomMatrix(N, m, k, A);
    RandomMatrix(N, k, 1, B);
    RandomMatrix(N, m, 1, C);

    switch (RandomInteger(3))
    {
        case 0: // ABC -> ABC
            break;
        case 1: // ABC -> ACB
            swap(B, C);
            A.transpose();
            break;
        case 2: // ABC -> BAC
            swap(A, B);
            A.transpose();
            B.transpose();
            C.transpose();
            break;
        case 3: // ABC -> CAB
            swap(A, B);
            swap(A, C);
            A.transpose();
            C.transpose();
            break;
    }
}

/*
 * Creates a random matrix outer product operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomGER(siz_t N, Matrix<T>& A,
                        Matrix<T>& B,
                        Matrix<T>& C)
{
    dim_t m = RandomInteger(1, (dim_t)sqrt(N));
    dim_t n = RandomInteger(1, (dim_t)sqrt(N));

    RandomMatrix(N, m, 1, A);
    RandomMatrix(N, 1, n, B);
    RandomMatrix(N, m, n, C);

    switch (RandomInteger(1))
    {
        case 0: // ABC -> ABC
            break;
        case 1: // ABC -> BAC
            swap(A, B);
            A.transpose();
            B.transpose();
            C.transpose();
            break;
    }
}

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void RandomContract(siz_t N, Tensor<T>& A, string& idx_A,
                             Tensor<T>& B, string& idx_B,
                             Tensor<T>& C, string& idx_C)
{
    gint_t ndim_A, ndim_B, ndim_C;
    gint_t ndim_AB, ndim_AC, ndim_BC;
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
void RandomWeight(siz_t N, Tensor<T>& A, string& idx_A,
                           Tensor<T>& B, string& idx_B,
                           Tensor<T>& C, string& idx_C)
{
    gint_t ndim_A, ndim_B, ndim_C;
    gint_t ndim_AC, ndim_BC;
    gint_t ndim_ABC;
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
void RandomOuterProd(siz_t N, Tensor<T>& A, string& idx_A,
                              Tensor<T>& B, string& idx_B,
                              Tensor<T>& C, string& idx_C)
{
    gint_t ndim_A, ndim_B, ndim_C;
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
void TestTBLIS(siz_t N)
{
    Matrix<T> A, B, C, D;
    ScatterMatrix<T> sA, sB, sC;

    for (int pass = 0;pass < 3;pass++)
    {
        switch (pass)
        {
            case 0: RandomGEMM(N, A, B, C); break;
            case 1: RandomGEMV(N, A, B, C); break;
            case 2: RandomGER (N, A, B, C); break;
        }

        T ref_val, calc_val;
        T scale = 10.0*RandomUnit<T>();
        Scalar<T> scale_obj(scale), res_obj;

        cout << endl;
        cout << "Testing TBLIS/" << (pass == 0 ? "GEMM" :
                                     pass == 1 ? "GEMV" :
                                                 "GER") << " (" << TypeName<T>() << "):" << endl;

        bool transa = A.is_transposed();
        bool transb = B.is_transposed();
        bool transc = C.is_transposed();

        cout << endl;
        cout << "m, n, k    = " << (transa ? A.width() : A.length()) << ", "
                                << (transb ? B.length() : B.width()) << ", "
                                << (transa ? A.length() : A.width()) << endl;
        cout << "rs_a, cs_a = " << A.row_stride() << ", " << A.col_stride() << endl;
        cout << "rs_b, cs_b = " << B.row_stride() << ", " << B.col_stride() << endl;
        cout << "rs_c, cs_c = " << C.row_stride() << ", " << C.col_stride() << endl;
        cout << "trans_a    = " << transa << endl;
        cout << "trans_b    = " << transb << endl;
        cout << "trans_c    = " << transc << endl;
        cout << endl;

        D = C;
        gemm_ref(scale, A, B, scale, D);
        bli_normfm(D, res_obj);
        ref_val = (T)res_obj;

        //D = C;
        //bli_gemm(scale_obj, A, B, scale_obj, D);
        //bli_normfm(D, res_obj);
        //ref_val = (T)res_obj;

        D = C;
        tblis_gemm(scale, A, B, scale, D);
        tblis_normfm(D, calc_val);

        passfail("REF", ref_val, calc_val);

        D = C;
        bli_gemm(scale_obj, A, B, scale_obj, D);
        bli_normfm(D, res_obj);
        ref_val = (T)res_obj;

        passfail("BLIS", ref_val, calc_val);

        D = C;
        sA.reset(A, SCATTER_NONE);
        sB.reset(B, SCATTER_NONE);
        sC.reset(D, SCATTER_NONE);

        tblis_gemm(scale, sA, sB, scale, sC);
        tblis_normfm(sC, calc_val);

        passfail("SCATTER_NONE", ref_val, calc_val);

        D = C;
        sA.reset(A, SCATTER_BOTH);
        sB.reset(B, SCATTER_BOTH);
        sC.reset(D, SCATTER_BOTH);

        tblis_gemm(scale, sA, sB, scale, sC);
        tblis_normfm(sC, calc_val);

        passfail("SCATTER_BOTH", ref_val, calc_val);

        D = C;
        sA.reset(A, (scatter_t)RandomInteger(0,3));
        sB.reset(B, (scatter_t)RandomInteger(0,3));
        sC.reset(D, (scatter_t)RandomInteger(0,3));

        tblis_gemm(scale, sA, sB, scale, sC);
        tblis_normfm(sC, calc_val);

        passfail("SCATTER_RANDOM", ref_val, calc_val);
    }
}

template <> void TestTBLIS<sComplex>(siz_t N) {}

template <> void TestTBLIS<dComplex>(siz_t N) {}

template <typename T>
void TestMult(siz_t N)
{
    Tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    cout << endl;
    cout << "Testing mult (" << TypeName<T>() << "):" << endl;

    RandomContract(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    impl_type = BLAS_BASED;
    D = C;
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D = C;
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("CONTRACT", ref_val, calc_val);

    RandomWeight(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    impl_type = BLAS_BASED;
    D = C;
    tensor_weight(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D = C;
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("WEIGHT", ref_val, calc_val);

    RandomOuterProd(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    impl_type = BLAS_BASED;
    D = C;
    tensor_outer_prod(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D = C;
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("OUTER_PROD", ref_val, calc_val);

    RandomMult(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    impl_type = BLAS_BASED;
    D = C;
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D = C;
    tensor_mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestContract(siz_t N)
{
    Tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    RandomContract(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing contract (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    impl_type = REFERENCE;
    D = C;
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = BLAS_BASED;
    D = C;
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    impl_type = REFERENCE;
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);

    impl_type = BLIS_BASED;
    D = C;
    tensor_contract(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    impl_type = REFERENCE;
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLIS", ref_val, calc_val);
}

template <typename T>
void TestWeight(siz_t N)
{
    Tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    RandomWeight(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing weight (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    impl_type = BLAS_BASED;
    D = C;
    tensor_weight(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D = C;
    tensor_weight(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestOuterProd(siz_t N)
{
    Tensor<T> A, B, C, D;
    string idx_A, idx_B, idx_C;

    RandomOuterProd(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "Testing outer prod (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << "len_C    = " << C.getLengths() << endl;
    cout << "stride_C = " << C.getStrides() << endl;
    cout << "idx_C    = " << idx_C << endl;
    cout << endl;

    T ref_val, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    impl_type = BLAS_BASED;
    D = C;
    tensor_outer_prod(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, ref_val);

    impl_type = REFERENCE;
    D = C;
    tensor_outer_prod(scale, A, idx_A, B, idx_B, scale, D, idx_C);
    tensor_reduce(REDUCE_NORM_2, D, idx_C, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestSum(siz_t N)
{
    Tensor<T> A, B, C;
    string idx_A, idx_B;

    T ref_val, calc_val, scale, sum_b;
    scale = 10.0*RandomUnit<T>();

    cout << endl;
    cout << "Testing sum (" << TypeName<T>() << "):" << endl;

    RandomTranspose(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    C = B;
    tensor_transpose(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C = B;
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("TRANSPOSE", ref_val, calc_val);

    RandomTrace(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    C = B;
    tensor_trace(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C = B;
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("TRACE", ref_val, calc_val);

    RandomReplicate(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    C = B;
    tensor_replicate(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C = B;
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("REPLICATE", ref_val, calc_val);

    RandomSum(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    inc_t sz = 1;
    for (gint_t i = 0;i < B.getDimension();i++)
    {
        bool found = false;
        for (gint_t j = 0;j < A.getDimension();j++)
        {
            if (idx_A[j] == idx_B[i])
            {
                found = true;
                break;
            }
        }
        if (!found) sz *= B.getLength(i);
    }

    impl_type = REFERENCE;
    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);
    tensor_reduce(REDUCE_SUM, B, idx_B, sum_b);
    tensor_sum(scale, A, idx_A, scale, B, idx_B);
    tensor_reduce(REDUCE_SUM, B, idx_B, calc_val);
    passfail("SUM", scale*(sz*ref_val+sum_b), calc_val);

    impl_type = BLAS_BASED;
    C = B;
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, ref_val);

    impl_type = REFERENCE;
    C = B;
    tensor_sum(scale, A, idx_A, scale, C, idx_B);
    tensor_reduce(REDUCE_NORM_2, C, idx_B, calc_val);

    passfail("BLAS", ref_val, calc_val);
}

template <typename T>
void TestTrace(siz_t N)
{
    Tensor<T> A, B;
    string idx_A, idx_B;

    RandomTrace(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing trace (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
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
void TestReplicate(siz_t N)
{
    Tensor<T> A, B;
    string idx_A, idx_B;

    RandomReplicate(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing replicate (" << TypeName<T>() << "):" << endl;
    cout << "len_A    = " << A.getLengths() << endl;
    cout << "stride_A = " << A.getStrides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.getLengths() << endl;
    cout << "stride_B = " << B.getStrides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, sum_b, calc_val, scale;
    scale = 10.0*RandomUnit<T>();

    inc_t sz = 1;
    for (gint_t i = 0;i < B.getDimension();i++)
    {
        bool found = false;
        for (gint_t j = 0;j < A.getDimension();j++)
        {
            if (idx_A[j] == idx_B[i])
            {
                found = true;
                break;
            }
        }
        if (!found) sz *= B.getLength(i);
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
void TestDot(siz_t N)
{
    Tensor<T> A, B;
    string idx_A, idx_B;

    RandomDot(N, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing dot (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.getLengths() << endl;
    cout << "stride = " << A.getStrides() << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, calc_val;

    tensor_transpose(T(1.0), A, idx_A, T(0.0), B, idx_B);
    B.conjugate();
    Normalize(A, idx_A);
    tensor_reduce(REDUCE_NORM_2, A, idx_A, ref_val);
    tensor_dot(A, idx_A, B, idx_B, calc_val);
    passfail("NRM2", ref_val*ref_val, calc_val);

    B = 1.0;
    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val);
    tensor_dot(A, idx_A, B, idx_B, calc_val);
    passfail("UNIT", ref_val, calc_val);

    B = 0.0;
    tensor_dot(A, idx_A, B, idx_B, calc_val);
    passfail("ZERO", T(0.0), calc_val);
}

template <typename T>
void TestTranspose(siz_t N)
{
    Tensor<T> A, B, C;
    string idx_A, idx_B;

    RandomTranspose(N, A, idx_A, B, idx_B);

    gint_t ndim = A.getDimension();
    string perm = permutation(idx_A, idx_B);

    cout << endl;
    cout << "Testing transpose (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.getLengths() << endl;
    cout << "stride = " << A.getStrides() << endl;
    cout << "perm   = " << perm << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, calc_val, scale;

    scale = 10.0*RandomUnit<T>();

    C = A;
    string idx_C = idx_A;
    tensor_reduce(REDUCE_NORM_2, A, idx_A, ref_val);
    tensor_transpose(T(1.0), A, idx_A, T(0.0), B, idx_B);
    tensor_transpose(scale, B, idx_B, scale, C, idx_C);
    tensor_reduce(REDUCE_NORM_2, C, idx_C, calc_val);
    passfail("INVERSE", T(2.0*std::abs(scale))*ref_val, calc_val);

    B = A;
    idx_B = idx_A;
    vector<dim_t> len_C(ndim);
    vector<inc_t> stride_C(ndim);
    do
    {
        for (gint_t i = 0;i < ndim;i++)
        {
            gint_t j; for (j = 0;j < ndim && idx_A[j] != perm[i];j++);
            idx_C[i] = idx_B[j];
            len_C[i] = B.getLength(j);
            stride_C[i] = B.getStride(j);
        }
        C.reset(ndim, len_C, stride_C);
        tensor_transpose(T(1.0), B, idx_B, T(0.0), C, idx_C);
        B = C;
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    tensor_reduce(REDUCE_NORM_2, C, idx_C, calc_val);
    passfail("CYCLE", ref_val, calc_val);
}

template <typename T>
void TestScale(siz_t N)
{
    Tensor<T> A;
    string idx_A = "abcdefgh";

    RandomTensor(N, A);
    idx_A.resize(A.getDimension());

    cout << endl;
    cout << "Testing scale (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.getLengths() << endl;
    cout << "stride = " << A.getStrides() << endl;
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
void TestReduce(siz_t N)
{
    Tensor<T> A;
    string idx_A = "abcdefgh";

    RandomTensor(N, A);
    idx_A.resize(A.getDimension());
    siz_t NA = A.getDataSize();

    cout << endl;
    cout << "Testing reduction (" << TypeName<T>() << "):" << endl;
    cout << "len    = " << A.getLengths() << endl;
    cout << "stride = " << A.getStrides() << endl;
    cout << endl;

    impl_type = REFERENCE;
    T ref_val, blas_val;
    inc_t ref_idx, blas_idx;

    T* data = A.getData();

    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val, ref_idx);
    blas_val = 0;
    for (siz_t i = 0;i < NA;i++)
    {
        blas_val += data[i];
    }
    passfail("REDUCE_SUM", ref_val, blas_val);

    tensor_reduce(REDUCE_SUM_ABS, A, idx_A, ref_val, ref_idx);
    blas_val = 0;
    for (siz_t i = 0;i < NA;i++)
    {
        blas_val += std::abs(data[i]);
    }
    passfail("REDUCE_SUM_ABS", ref_val, blas_val);

    tensor_reduce(REDUCE_MAX, A, idx_A, ref_val, ref_idx);
    blas_val = data[0];
    blas_idx = 0;
    for (siz_t i = 0;i < NA;i++)
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
    for (siz_t i = 0;i < NA;i++)
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
    for (siz_t i = 0;i < NA;i++)
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
    for (siz_t i = 0;i < NA;i++)
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
    for (siz_t i = 0;i < NA;i++)
    {
        blas_val += norm2(data[i]);
    }
    blas_val = sqrt(real(blas_val));
    passfail("REDUCE_NORM_2", ref_val, blas_val);

    A = 1;
    tensor_reduce(REDUCE_SUM, A, idx_A, ref_val, ref_idx);
    blas_val = 1;
    for (int i = 0;i < A.getDimension();i++) blas_val *= A.getLength(i);
    passfail("COUNT", ref_val, blas_val);
}

template <typename T>
void Test(siz_t N_in_bytes, gint_t R)
{
    siz_t N = N_in_bytes/sizeof(T);

    for (gint_t i = 0;i < R;i++) TestTBLIS<T>(N);

    for (gint_t i = 0;i < R;i++) TestReduce<T>(N);
    for (gint_t i = 0;i < R;i++) TestScale<T>(N);
    for (gint_t i = 0;i < R;i++) TestTranspose<T>(N);
    for (gint_t i = 0;i < R;i++) TestDot<T>(N);
    for (gint_t i = 0;i < R;i++) TestReplicate<T>(N);
    for (gint_t i = 0;i < R;i++) TestTrace<T>(N);
    for (gint_t i = 0;i < R;i++) TestSum<T>(N);
    for (gint_t i = 0;i < R;i++) TestOuterProd<T>(N);
    for (gint_t i = 0;i < R;i++) TestWeight<T>(N);
    for (gint_t i = 0;i < R;i++) TestContract<T>(N);
    for (gint_t i = 0;i < R;i++) TestMult<T>(N);
}

int main(int argc, char **argv)
{
    siz_t N = 1024*1024;
    gint_t R = 10;
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
                abort();
                break;
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    engine.seed(seed);

    Test<   float>(N, R);
    Test<  double>(N, R);
    Test<sComplex>(N, R);
    Test<dComplex>(N, R);

    tblis_finalize();

    return 0;
}
