#include <algorithm>
#include <limits>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <iomanip>
#include <map>

#include "tblis.h"

#include "internal/3t/mult.hpp"
#include "util/random.hpp"
#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/iostream.hpp"

using namespace std;
using namespace stl_ext;
using namespace tblis;
using namespace tblis::internal;

constexpr int ulp_factor = 32;

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
vector<len_type> group_size(const matrix<len_type>& len, const T& idx, const T& choose)
{
    unsigned nirrep = len.length(1);
    matrix<len_type> sublen({(len_type)choose.size(), nirrep});

    for (unsigned i = 0;i < choose.size();i++)
    {
        for (unsigned j = 0;j < idx.size();j++)
        {
            if (choose[i] == idx[j])
            {
                sublen[i] = len[j];
            }
        }
    }

    vector<len_type> size(nirrep);
    for (unsigned i = 0;i < nirrep;i++)
    {
        size[i] = dpd_varray<double>::size(i, sublen);
    }

    return size;
}

template <typename T>
double ceil2(T x)
{
    return nearbyint(pow(2.0, max(0.0, ceil(log2((double)std::abs(x))))));
}

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
void passfail(const string& label, stride_type ia, stride_type ib, T a, U b, double ulps)
{
    auto c = real(std::abs(a-b));
    decltype(c) epsilon(ulps*numeric_limits<decltype(c)>::epsilon());
    bool pass = c < max(numeric_limits<decltype(c)>::min(), epsilon) && ia == ib;

    cout << label << ": ";
    if (pass)
    {
        cout << "pass" << endl;
    }
    else
    {
        cout << "fail" << endl;
        cout << scientific << setprecision(15);
        cout << a << " " << ia << endl;
        cout << b << " " << ib << endl;
        cout << c << " > " << max(numeric_limits<decltype(c)>::min(), epsilon) << endl;
        ::abort();
    }
}

template <typename T, typename U>
void passfail(const string& label, T a, U b, double ulps)
{
    passfail(label, 0, 0, a, b, ulps);
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
    miterator<2> it(t.lengths(), t.strides());
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

    //m = 3;
    //n = 3;
    //k = 3;

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
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, const vector<len_type>& len_min, vector<len_type>& len)
{
    vector<len_type> len_max = random_product_constrained_sequence<len_type>(d, N, len_min);

    len.resize(d);
    for (unsigned i = 0;i < d;i++)
    {
        len[i] = (len_min[i] > 0 ? len_min[i] : random_number<len_type>(1, len_max[i]));
    }
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4 and N/2. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, vector<len_type>& len)
{
    random_lengths(N, d, vector<len_type>(d), len);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
void random_lengths(stride_type N, vector<len_type>& len)
{
    random_lengths(N, random_number(1,8), len);
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void randomize_tensor(varray<T>& t)
{
    t.for_each_element(
    [](T& e, const vector<len_type>&)
    {
        e = random_unit<T>();
    });
}

template <typename T>
void random_tensor(stride_type N, unsigned d, const vector<len_type>& len_min, varray<T>& A)
{
    vector<len_type> len_A;
    random_lengths(N, d, len_min, len_A);
    A.reset(len_A);
    randomize_tensor(A);
}

template <typename T>
void random_tensor(stride_type N, unsigned d, varray<T>& A)
{
    random_tensor(N, d, vector<len_type>(d), A);
}

template <typename T>
void random_tensor(stride_type N, varray<T>& A)
{
    random_tensor(N, random_number(1,8), A);
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
template <typename T>
void randomize_dpd_tensor(dpd_varray<T>& t)
{
    t.for_each_element(
    [](T& e, const vector<unsigned>&, const vector<len_type>&)
    {
        e = random_unit<T>();
    });
}

template <typename T>
void random_dpd_tensor(stride_type N, unsigned d, unsigned nirrep, const vector<len_type>& len_min, dpd_varray<T>& A)
{
    unsigned irrep_A;
    vector<vector<len_type>> len_A(d);

    do
    {
        irrep_A = random_number(nirrep-1);

        vector<len_type> len_A_;
        random_lengths(nirrep*N, d, len_min, len_A_);

        for (unsigned i = 0;i < d;i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);
    }
    while (A.size(irrep_A, len_A) == 0);

    A.reset(irrep_A, nirrep, len_A);
    randomize_dpd_tensor(A);
}

template <typename T>
void random_dpd_tensor(stride_type N, unsigned d, unsigned nirrep, dpd_varray<T>& A)
{
    random_dpd_tensor(N, d, nirrep, vector<len_type>(d), A);
}

template <typename T>
void random_dpd_tensor(stride_type N, dpd_varray<T>& A)
{
    random_dpd_tensor(N, random_number(1,8), 1 << random_number(3), A);
}

void random_lengths(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    vector<len_type>& len_A, vector<label_type>& idx_A,
                    vector<len_type>& len_B, vector<label_type>& idx_B)
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

    random_lengths(N, ndim_A, len_A);

    vector<len_type> min_B(ndim_B);
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

void random_lengths(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    vector<len_type>& len_A, vector<label_type>& idx_A,
                    vector<len_type>& len_B, vector<label_type>& idx_B,
                    vector<len_type>& len_C, vector<label_type>& idx_C)
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
        range<label_type>('a', static_cast<char>('a'+ndim_A_only+ndim_B_only+ndim_C_only+
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
        random_lengths(N, ndim_A, len_A);

        vector<len_type> min_B(ndim_B);
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
        vector<len_type> min_C(ndim_C);
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
                    unsigned ndim_A_only, unsigned ndim_B_only,
                    unsigned ndim_AB,
                    varray<T>& A, vector<label_type>& idx_A,
                    varray<T>& B, vector<label_type>& idx_B)
{
    vector<len_type> len_A, len_B;

    random_lengths(N, ndim_A_only, ndim_B_only, ndim_AB,
                   len_A, idx_A, len_B, idx_B);

    A.reset(len_A);
    B.reset(len_B);

    randomize_tensor(A);
    randomize_tensor(B);
}

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    varray<T>& A, vector<label_type>& idx_A,
                    varray<T>& B, vector<label_type>& idx_B,
                    varray<T>& C, vector<label_type>& idx_C)
{
    vector<len_type> len_A, len_B, len_C;

    random_lengths(N, ndim_A_only, ndim_B_only, ndim_C_only,
                   ndim_AB, ndim_AC, ndim_BC, ndim_ABC,
                   len_A, idx_A, len_B, idx_B, len_C, idx_C);

    A.reset(len_A);
    B.reset(len_B);
    C.reset(len_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
}

template <typename T>
void random_dpd_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    dpd_varray<T>& A, vector<label_type>& idx_A,
                    dpd_varray<T>& B, vector<label_type>& idx_B)
{
    unsigned nirrep, irrep_A, irrep_B;
    vector<vector<len_type>> len_A, len_B;

    do
    {
        nirrep = 1 << random_number(3);
        irrep_A = irrep_B = random_number(nirrep-1);

        vector<len_type> len_A_, len_B_;

        random_lengths(nirrep*N, ndim_A_only, ndim_B_only, ndim_AB,
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
    while (A.size(irrep_A, len_A) == 0 ||
           B.size(irrep_B, len_B) == 0);

    A.reset(irrep_A, nirrep, len_A);
    B.reset(irrep_B, nirrep, len_B);

    randomize_dpd_tensor(A);
    randomize_dpd_tensor(B);
}

template <typename T>
void random_dpd_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_C_only,
                    unsigned ndim_AB, unsigned ndim_AC, unsigned ndim_BC,
                    unsigned ndim_ABC,
                    dpd_varray<T>& A, vector<label_type>& idx_A,
                    dpd_varray<T>& B, vector<label_type>& idx_B,
                    dpd_varray<T>& C, vector<label_type>& idx_C)
{
    unsigned nirrep, irrep_A, irrep_B, irrep_C;
    vector<vector<len_type>> len_A, len_B, len_C;

    do
    {
        nirrep = 1 << random_number(3);
        irrep_A = random_number(nirrep-1);
        irrep_B = random_number(nirrep-1);
        irrep_C = irrep_A^irrep_B;

        vector<len_type> len_A_, len_B_, len_C_;

        random_lengths(nirrep*N, ndim_A_only, ndim_B_only, ndim_C_only,
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
    while (A.size(irrep_A, len_A) == 0 ||
           B.size(irrep_B, len_B) == 0 ||
           C.size(irrep_C, len_C) == 0);

    A.reset(irrep_A, nirrep, len_A);
    B.reset(irrep_B, nirrep, len_B);
    C.reset(irrep_C, nirrep, len_C);

    randomize_dpd_tensor(A);
    randomize_dpd_tensor(B);
    randomize_dpd_tensor(C);
}

/*
 * Creates a random tensor addmation operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_add(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                          varray<T>& B, vector<label_type>& idx_B)
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
void random_trace(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                            varray<T>& B, vector<label_type>& idx_B)
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
void random_replicate(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                                varray<T>& B, vector<label_type>& idx_B)
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
void random_transpose(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                                varray<T>& B, vector<label_type>& idx_B)
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
void random_dot(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                          varray<T>& B, vector<label_type>& idx_B)
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
void random_mult(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                           varray<T>& B, vector<label_type>& idx_B,
                           varray<T>& C, vector<label_type>& idx_C)
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
           ndim_A_only == ndim_A ||
           ndim_B_only == ndim_B ||
           ndim_C_only == ndim_C ||
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
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_contract(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                              varray<T>& B, vector<label_type>& idx_B,
                              varray<T>& C, vector<label_type>& idx_C)
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
void random_weight(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                            varray<T>& B, vector<label_type>& idx_B,
                            varray<T>& C, vector<label_type>& idx_C)
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
void random_outer_prod(stride_type N, varray<T>& A, vector<label_type>& idx_A,
                                 varray<T>& B, vector<label_type>& idx_B,
                                 varray<T>& C, vector<label_type>& idx_C)
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

/*
 * Creates a random tensor addmation operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_dpd_add(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                   dpd_varray<T>& B, vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);
    unsigned ndim_B = random_number(1,8);

    unsigned ndim_AB = random_number(0u, min(ndim_A,ndim_B));
    unsigned ndim_A_only = ndim_A-ndim_AB;
    unsigned ndim_B_only = ndim_B-ndim_AB;

    random_dpd_tensors(N,
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
void random_dpd_trace(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                     dpd_varray<T>& B, vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);
    unsigned ndim_B = random_number(1,8);

    if (ndim_A < ndim_B) swap(ndim_A, ndim_B);

    random_dpd_tensors(N,
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
void random_dpd_replicate(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                         dpd_varray<T>& B, vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);
    unsigned ndim_B = random_number(1,8);

    if (ndim_B < ndim_A) swap(ndim_A, ndim_B);

    random_dpd_tensors(N,
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
void random_dpd_transpose(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                         dpd_varray<T>& B, vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);

    random_dpd_tensors(N,
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
void random_dpd_dot(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                   dpd_varray<T>& B, vector<label_type>& idx_B)
{
    unsigned ndim_A = random_number(1,8);

    random_dpd_tensors(N,
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
void random_dpd_mult(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                    dpd_varray<T>& B, vector<label_type>& idx_B,
                                    dpd_varray<T>& C, vector<label_type>& idx_C)
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
           ndim_A_only == ndim_A ||
           ndim_B_only == ndim_B ||
           ndim_C_only == ndim_C ||
           ((ndim_A-ndim_A_only)+
            (ndim_B-ndim_B_only)-
            (ndim_C-ndim_C_only)-ndim_ABC)%2 != 0);

    random_dpd_tensors(N,
                       ndim_A_only, ndim_B_only, ndim_C_only,
                       ndim_AB, ndim_AC, ndim_BC,
                       ndim_ABC,
                       A, idx_A,
                       B, idx_B,
                       C, idx_C);
}

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_dpd_contract(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                        dpd_varray<T>& B, vector<label_type>& idx_B,
                                        dpd_varray<T>& C, vector<label_type>& idx_C)
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

    random_dpd_tensors(N,
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
void random_dpd_weight(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                      dpd_varray<T>& B, vector<label_type>& idx_B,
                                      dpd_varray<T>& C, vector<label_type>& idx_C)
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

    random_dpd_tensors(N,
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
void random_dpd_outer_prod(stride_type N, dpd_varray<T>& A, vector<label_type>& idx_A,
                                          dpd_varray<T>& B, vector<label_type>& idx_B,
                                          dpd_varray<T>& C, vector<label_type>& idx_C)
{
    unsigned ndim_A, ndim_B, ndim_C;
    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = ndim_A+ndim_B;
    }
    while (ndim_C > 8);

    random_dpd_tensors(N,
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
    matrix<T> A, B, C, D, E;

    for (int pass = 0;pass < 3;pass++)
    {
        switch (pass)
        {
            case 0: random_gemm(N, A, B, C); break;
            case 1: random_gemv(N, A, B, C); break;
            case 2: random_ger (N, A, B, C); break;
        }

        T scale(10.0*random_unit<T>());

        cout << endl;
        cout << "Testing TBLIS/" << (pass == 0 ? "GEMM" :
                                     pass == 1 ? "GEMV" :
                                                 "GER") << " (" << type_name<T>() << "):" << endl;

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);

        cout << endl;
        cout << "m, n, k    = " << m << ", " << n << ", " << k << endl;
        cout << "rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1) << endl;
        cout << "rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1) << endl;
        cout << "rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1) << endl;
        cout << endl;

        D.reset(C);
        gemm_ref<T>(scale, A, B, scale, D);

        E.reset(C);
        mult<T>(scale, A, B, scale, E);

        add<T>(T(-1), D, T(1), E);
        T error = reduce<T>(REDUCE_NORM_2, E).first;

        passfail("REF", error, 0, ulp_factor*ceil2(scale*m*n*k));
    }
}

template <typename T>
void test_mult(stride_type N)
{
    varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

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

    auto idx_AB = exclusion(intersection(idx_A, idx_B), idx_C);
    auto idx_A_only = exclusion(idx_A, idx_B, idx_C);
    auto idx_B_only = exclusion(idx_B, idx_A, idx_C);

    auto neps = ceil2(prod(select_from(A.lengths(), idx_A, idx_A_only))*
                      prod(select_from(A.lengths(), idx_A, idx_AB))*
                      prod(select_from(B.lengths(), idx_B, idx_B_only))*
                      prod(C.lengths()));

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_contract(stride_type N)
{
    varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    T scale(10.0*random_unit<T>());

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

    auto idx_AB = intersection(idx_A, idx_B);

    auto neps = ceil2(prod(select_from(A.lengths(), idx_A, idx_AB))*
                      prod(C.lengths()));

    impl = REFERENCE;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = BLAS_BASED;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));

    impl = BLIS_BASED;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLIS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_weight(stride_type N)
{
    varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

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

    auto neps = ceil2(prod(C.lengths()));

    T scale(10.0*random_unit<T>());

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_outer_prod(stride_type N)
{
    varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

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

    auto neps = ceil(prod(C.lengths()));

    T scale(10.0*random_unit<T>());

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_add(stride_type N)
{
    varray<T> A, B, C;
    vector<label_type> idx_A, idx_B;

    T scale(10.0*random_unit<T>());

    cout << endl;
    cout << "Testing add (" << type_name<T>() << "):" << endl;

    random_add(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    auto idx_B_only = exclusion(idx_B, idx_A);
    stride_type NB = prod(select_from(B.lengths(), idx_B, idx_B_only));
    auto neps = prod(A.lengths())*NB;

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(NB*ref_val+add_b), calc_val, ulp_factor*ceil2(neps*scale));
}

template <typename T>
void test_trace(stride_type N)
{
    varray<T> A, B;
    vector<label_type> idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing trace (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(ref_val+add_b), calc_val, ulp_factor*ceil2(neps*scale));
}

template <typename T>
void test_replicate(stride_type N)
{
    varray<T> A, B;
    vector<label_type> idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing replicate (" << type_name<T>() << "):" << endl;
    cout << "len_A    = " << A.lengths() << endl;
    cout << "stride_A = " << A.strides() << endl;
    cout << "idx_A    = " << idx_A << endl;
    cout << "len_B    = " << B.lengths() << endl;
    cout << "stride_B = " << B.strides() << endl;
    cout << "idx_B    = " << idx_B << endl;
    cout << endl;

    auto idx_B_only = exclusion(idx_B, idx_A);
    stride_type NB = prod(select_from(B.lengths(), idx_B, idx_B_only));
    auto neps = prod(B.lengths());

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(NB*ref_val+add_b), calc_val, ulp_factor*ceil2(neps*scale));

    ref_val = reduce<T>(REDUCE_NORM_1, A, idx_A.data()).first;
    add<T>(scale, A, idx_A.data(), T(0.0), B, idx_B.data());
    calc_val = reduce<T>(REDUCE_NORM_1, B, idx_B.data()).first;
    passfail("NRM1", std::abs(scale)*NB*ref_val, calc_val, ulp_factor*ceil2(neps*scale));
}

template <typename T>
void test_dot(stride_type N)
{
    varray<T> A, B;
    vector<label_type> idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing dot (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    auto neps = prod(A.lengths());

    add<T>(T(1.0), A, idx_A.data(), T(0.0), B, idx_B.data());
    B.for_each_element([](T& e) { e = tblis::conj(e); });
    T ref_val = reduce<T>(REDUCE_NORM_2, A, idx_A.data()).first;
    T calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    passfail("NRM2", ref_val*ref_val, calc_val, ulp_factor*ceil2(neps));

    B = T(1);
    ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    passfail("UNIT", ref_val, calc_val, ulp_factor*ceil2(neps));

    B = T(0);
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    passfail("ZERO", calc_val, 0, ulp_factor*ceil2(neps));
}

template <typename T>
void test_transpose(stride_type N)
{
    varray<T> A, B, C;
    vector<label_type> idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    unsigned ndim = A.dimension();
    vector<unsigned> perm = permutation(ndim, idx_A.data(), idx_B.data());

    cout << endl;
    cout << "Testing transpose (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << "perm   = " << perm << endl;
    cout << endl;

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    C.reset(A);
    add<T>(T(1), A, idx_A.data(), T(0), B, idx_B.data());
    add<T>(scale, B, idx_B.data(), scale, C, idx_A.data());

    add<T>(-2*scale, A, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    passfail("INVERSE", error, 0, ulp_factor*ceil2(2*scale*neps));

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
        add<T>(T(1), B, idx_B.data(), T(0), C, idx_C.data());
        B.reset(C);
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    add<T>(T(-1), A, idx_A.data(), T(1), C, idx_A.data());
    error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    passfail("CYCLE", error, 0, ulp_factor*ceil2(neps));
}

template <typename T>
void test_scale(stride_type N)
{
    varray<T> A;

    random_tensor(100, A);
    vector<label_type> idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    cout << endl;
    cout << "Testing scale (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    auto neps = prod(A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    tblis::scale<T>(scale, A, idx_A.data());
    T calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    passfail("RANDOM", ref_val, calc_val/scale, ulp_factor*ceil2(neps));

    tblis::scale<T>(T(1.0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    passfail("UNIT", ref_val, calc_val/scale, ulp_factor*ceil2(neps));

    tblis::scale<T>(T(0.0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    passfail("ZERO", calc_val, 0, ulp_factor*ceil2(neps));
}

template <typename T>
void test_reduce(stride_type N)
{
    varray<T> A;

    random_tensor(100, A);
    vector<label_type> idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    cout << endl;
    cout << "Testing reduction (" << type_name<T>() << "):" << endl;
    cout << "len    = " << A.lengths() << endl;
    cout << "stride = " << A.strides() << endl;
    cout << endl;

    stride_type NA = prod(A.lengths());

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        blas_val += data[i];
    }
    passfail("REDUCE_SUM", ref_val, blas_val, ulp_factor*ceil2(NA));

    reduce<T>(REDUCE_SUM_ABS, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        blas_val += std::abs(data[i]);
    }
    passfail("REDUCE_SUM_ABS", ref_val, blas_val, ulp_factor*ceil2(NA));

    reduce<T>(REDUCE_MAX, A, idx_A.data(), ref_val, ref_idx);
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
    passfail("REDUCE_MAX", ref_idx, blas_idx, ref_val, blas_val, 4);

    reduce<T>(REDUCE_MAX_ABS, A, idx_A.data(), ref_val, ref_idx);
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
    passfail("REDUCE_MAX_ABS", ref_idx, blas_idx, ref_val, blas_val, 4);

    /*
    reduce<T>(REDUCE_MIN, A, idx_A.data(), ref_val, ref_idx);
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

    reduce<T>(REDUCE_MIN_ABS, A, idx_A.data(), ref_val, ref_idx);
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

    reduce<T>(REDUCE_NORM_2, A, idx_A.data(), ref_val, ref_idx);
    blas_val = 0;
    for (stride_type i = 0;i < NA;i++)
    {
        blas_val += norm2(data[i]);
    }
    blas_val = sqrt(real(blas_val));
    passfail("REDUCE_NORM_2", ref_val, blas_val, ulp_factor*ceil2(NA));

    A = T(1);
    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    passfail("COUNT", ref_val, NA, ulp_factor*ceil2(NA));
}

template <typename T>
void test_dpd_mult(stride_type N)
{
    dpd_varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    cout << endl;
    cout << "Testing DPD mult (" << type_name<T>() << "):" << endl;

    random_dpd_mult(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << "irrep_C = " << C.irrep() << endl;
    cout << "len_C   = " << endl << C.lengths() << endl;
    cout << "idx_C   = " << idx_C << endl;
    cout << endl;

    auto idx_ABC = intersection(idx_A, idx_B, idx_C);
    auto idx_AB = exclusion(intersection(idx_A, idx_B), idx_C);
    auto idx_AC = exclusion(intersection(idx_A, idx_C), idx_B);
    auto idx_BC = exclusion(intersection(idx_B, idx_C), idx_A);
    auto idx_A_only = exclusion(idx_A, idx_B, idx_C);
    auto idx_B_only = exclusion(idx_B, idx_A, idx_C);
    auto idx_C_only = exclusion(idx_C, idx_A, idx_B);

    auto size_ABC = group_size(A.lengths(), idx_A, idx_ABC);
    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);
    auto size_A = group_size(A.lengths(), idx_A, idx_A_only);
    auto size_B = group_size(B.lengths(), idx_B, idx_B_only);
    auto size_C = group_size(C.lengths(), idx_C, idx_C_only);

    unsigned nirrep = A.num_irreps();
    stride_type nmult = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        for (unsigned irrep_AC = 0;irrep_AC < nirrep;irrep_AC++)
        {
            for (unsigned irrep_BC = 0;irrep_BC < nirrep;irrep_BC++)
            {
                for (unsigned irrep_ABC = 0;irrep_ABC < nirrep;irrep_ABC++)
                {
                    unsigned irrep_A = A.irrep()^irrep_AB^irrep_AC^irrep_ABC;
                    unsigned irrep_B = B.irrep()^irrep_AB^irrep_BC^irrep_ABC;
                    unsigned irrep_C = C.irrep()^irrep_AC^irrep_BC^irrep_ABC;

                    nmult += size_ABC[irrep_ABC]*
                             size_AB[irrep_AB]*
                             size_AC[irrep_AC]*
                             size_BC[irrep_BC]*
                             size_A[irrep_A]*
                             size_B[irrep_B]*
                             size_C[irrep_C];
                }
            }
        }
    }

    auto neps = ceil2(nmult);

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_dpd_contract(stride_type N)
{
    dpd_varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    cout << endl;
    cout << "Testing DPD contract (" << type_name<T>() << "):" << endl;

    random_dpd_contract(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << "irrep_C = " << C.irrep() << endl;
    cout << "len_C   = " << endl << C.lengths() << endl;
    cout << "idx_C   = " << idx_C << endl;
    cout << endl;

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_AC = intersection(idx_A, idx_C);
    auto idx_BC = intersection(idx_B, idx_C);

    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);

    unsigned nirrep = A.num_irreps();
    stride_type nmult = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_AC = A.irrep()^irrep_AB;
        unsigned irrep_BC = B.irrep()^irrep_AB;

        nmult += size_AB[irrep_AB]*
                 size_AC[irrep_AC]*
                 size_BC[irrep_BC];
    }

    auto neps = ceil2(nmult);

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_dpd_weight(stride_type N)
{
    dpd_varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    cout << endl;
    cout << "Testing DPD weight (" << type_name<T>() << "):" << endl;

    random_dpd_weight(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << "irrep_C = " << C.irrep() << endl;
    cout << "len_C   = " << endl << C.lengths() << endl;
    cout << "idx_C   = " << idx_C << endl;
    cout << endl;

    auto idx_ABC = intersection(idx_A, idx_B, idx_C);
    auto idx_AC = exclusion(intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = exclusion(intersection(idx_B, idx_C), idx_ABC);

    auto size_ABC = group_size(A.lengths(), idx_A, idx_ABC);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);

    unsigned nirrep = A.num_irreps();
    stride_type nmult = 0;
    for (unsigned irrep_ABC = 0;irrep_ABC < nirrep;irrep_ABC++)
    {
        unsigned irrep_AC = A.irrep()^irrep_ABC;
        unsigned irrep_BC = B.irrep()^irrep_ABC;

        nmult += size_ABC[irrep_ABC]*
                 size_AC[irrep_AC]*
                 size_BC[irrep_BC];
    }

    auto neps = ceil2(nmult);

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_dpd_outer_prod(stride_type N)
{
    dpd_varray<T> A, B, C, D, E;
    vector<label_type> idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    cout << endl;
    cout << "Testing DPD outer prod (" << type_name<T>() << "):" << endl;

    random_dpd_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    cout << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << "irrep_C = " << C.irrep() << endl;
    cout << "len_C   = " << endl << C.lengths() << endl;
    cout << "idx_C   = " << idx_C << endl;
    cout << endl;

    auto neps = ceil2(group_size(C.lengths(), idx_C, idx_C)[C.irrep()]);

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    passfail("BLAS", error, 0, ulp_factor*ceil2(scale*neps));
}

template <typename T>
void test_dpd_add(stride_type N)
{
    dpd_varray<T> A, B, C;
    vector<label_type> idx_A, idx_B;

    T scale(10.0*random_unit<T>());

    random_dpd_add(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing DPD add (" << type_name<T>() << "):" << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << endl;

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_A_only = exclusion(idx_A, idx_B);
    auto idx_B_only = exclusion(idx_B, idx_A);

    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_A = group_size(A.lengths(), idx_A, idx_A_only);
    auto size_B = group_size(B.lengths(), idx_B, idx_B_only);

    unsigned nirrep = A.num_irreps();
    stride_type nadd = 0;
    stride_type NB_ = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_A = A.irrep()^irrep_AB;
        unsigned irrep_B = B.irrep()^irrep_AB;

        nadd += size_AB[irrep_AB]*
                size_A[irrep_A]*
                size_B[irrep_B];
        NB_ += size_B[irrep_B];
    }
    double NB = NB_/nirrep;

    auto neps = ceil2(nadd);

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(NB*ref_val+add_b), calc_val, ulp_factor*ceil2(neps*scale));
}

template <typename T>
void test_dpd_trace(stride_type N)
{
    dpd_varray<T> A, B;
    vector<label_type> idx_A, idx_B;

    random_dpd_trace(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing DPD trace (" << type_name<T>() << "):" << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << endl;

    auto neps = ceil2(group_size(A.lengths(), idx_A, idx_A)[A.irrep()]);

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(ref_val+add_b), calc_val, ulp_factor*ceil2(neps*scale));
}

template <typename T>
void test_dpd_replicate(stride_type N)
{
    dpd_varray<T> A, B;
    vector<label_type> idx_A, idx_B;

    random_dpd_replicate(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing DPD replicate (" << type_name<T>() << "):" << endl;
    cout << "irrep_A = " << A.irrep() << endl;
    cout << "len_A   = " << endl << A.lengths() << endl;
    cout << "idx_A   = " << idx_A << endl;
    cout << "irrep_B = " << B.irrep() << endl;
    cout << "len_B   = " << endl << B.lengths() << endl;
    cout << "idx_B   = " << idx_B << endl;
    cout << endl;

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_B_only = exclusion(idx_B, idx_A);

    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_B = group_size(B.lengths(), idx_B, idx_B_only);

    unsigned irrep_AB = A.irrep();
    unsigned irrep_B = B.irrep()^irrep_AB;
    stride_type nadd = size_AB[irrep_AB]*size_B[irrep_B];
    stride_type NB = size_B[irrep_B];

    auto neps = ceil2(nadd);

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    passfail("SUM", scale*(NB*ref_val+add_b), calc_val, ulp_factor*ceil2(neps*scale));

    ref_val = reduce<T>(REDUCE_NORM_1, A, idx_A.data()).first;
    add<T>(scale, A, idx_A.data(), T(0.0), B, idx_B.data());
    calc_val = reduce<T>(REDUCE_NORM_1, B, idx_B.data()).first;
    passfail("NRM1", std::abs(scale)*NB*ref_val, calc_val, ulp_factor*ceil2(neps*scale));
}

template <typename T>
void test_dpd_dot(stride_type N)
{
    dpd_varray<T> A, B;
    vector<label_type> idx_A, idx_B;

    random_dpd_dot(1000, A, idx_A, B, idx_B);

    cout << endl;
    cout << "Testing DPD dot (" << type_name<T>() << "):" << endl;
    cout << "irrep = " << A.irrep() << endl;
    cout << "len   = " << endl << A.lengths() << endl;
    cout << endl;

    auto neps = ceil2(group_size(A.lengths(), idx_A, idx_A)[A.irrep()]);

    add<T>(T(1.0), A, idx_A.data(), T(0.0), B, idx_B.data());
    B.for_each_element([](T& e) { e = tblis::conj(e); });
    T ref_val = reduce<T>(REDUCE_NORM_2, A, idx_A.data()).first;
    T calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    passfail("NRM2", ref_val*ref_val, calc_val, ulp_factor*ceil2(neps));

    B = T(1);
    ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    passfail("UNIT", ref_val, calc_val, ulp_factor*ceil2(neps));

    B = T(0);
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    passfail("ZERO", calc_val, 0, ulp_factor*ceil2(neps));
}

template <typename T>
void test_dpd_transpose(stride_type N)
{
    dpd_varray<T> A, B, C;
    vector<label_type> idx_A, idx_B;

    random_dpd_transpose(1000, A, idx_A, B, idx_B);

    unsigned ndim = A.dimension();
    vector<unsigned> perm = permutation(ndim, idx_A.data(), idx_B.data());

    cout << endl;
    cout << "Testing DPD transpose (" << type_name<T>() << "):" << endl;
    cout << "irrep = " << A.irrep() << endl;
    cout << "len   = " << endl << A.lengths() << endl;
    cout << "perm  = " << perm << endl;
    cout << endl;

    auto neps = ceil2(group_size(A.lengths(), idx_A, idx_A)[A.irrep()]);

    T scale(10.0*random_unit<T>());

    C.reset(A);
    add<T>(T(1), A, idx_A.data(), T(0), B, idx_B.data());
    add<T>(scale, B, idx_B.data(), scale, C, idx_A.data());

    add<T>(-2*scale, A, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    passfail("INVERSE", error, 0, ulp_factor*ceil2(2*scale*neps));

    B.reset(A);
    idx_B = idx_A;
    vector<label_type> idx_C(ndim);
    vector<vector<len_type>> len_C(ndim, vector<len_type>(A.num_irreps()));
    do
    {
        for (unsigned i = 0;i < ndim;i++)
        {
            unsigned j; for (j = 0;j < ndim && idx_A[j] != static_cast<label_type>(perm[i]+'a');j++) continue;
            idx_C[i] = idx_B[j];
            for (unsigned k = 0;k < B.num_irreps();k++)
                len_C[i][k] = B.length(j, k);
        }
        C.reset(B.irrep(), B.num_irreps(), len_C);
        add<T>(T(1), B, idx_B.data(), T(0), C, idx_C.data());
        B.reset(C);
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    add<T>(T(-1), A, idx_A.data(), T(1), C, idx_A.data());
    error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    passfail("CYCLE", error, 0, ulp_factor*ceil2(neps));
}

template <typename T>
void test_dpd_scale(stride_type N)
{
    dpd_varray<T> A;

    random_dpd_tensor(100, A);
    vector<label_type> idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    cout << endl;
    cout << "Testing DPD scale (" << type_name<T>() << "):" << endl;
    cout << "irrep = " << A.irrep() << endl;
    cout << "len   = " << endl << A.lengths() << endl;
    cout << endl;

    auto neps = ceil2(group_size(A.lengths(), idx_A, idx_A)[A.irrep()]);

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    tblis::scale<T>(scale, A, idx_A.data());
    T calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    passfail("RANDOM", ref_val, calc_val/scale, ulp_factor*ceil2(neps));

    tblis::scale<T>(T(1.0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    passfail("UNIT", ref_val, calc_val/scale, ulp_factor*ceil2(neps));

    tblis::scale<T>(T(0.0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    passfail("ZERO", calc_val, 0, ulp_factor*ceil2(neps));
}

template <typename T>
void test_dpd_reduce(stride_type N)
{
    dpd_varray<T> A;

    random_dpd_tensor(100, A);
    vector<label_type> idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    cout << endl;
    cout << "Testing DPD reduction (" << type_name<T>() << "):" << endl;
    cout << "irrep = " << A.irrep() << endl;
    cout << "len   = " << endl << A.lengths() << endl;
    cout << endl;

    auto NA = group_size(A.lengths(), idx_A, idx_A)[A.irrep()];

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

    map<reduce_t, string> ops =
    {
     {REDUCE_SUM, "REDUCE_SUM"},
     {REDUCE_SUM_ABS, "REDUCE_SUM_ABS"},
     {REDUCE_MAX, "REDUCE_MAX"},
     {REDUCE_MAX_ABS, "REDUCE_MAX_ABS"},
     {REDUCE_MIN, "REDUCE_MIN"},
     {REDUCE_MIN_ABS, "REDUCE_MIN_ABS"},
     {REDUCE_NORM_2, "REDUCE_NORM_2"}
    };

    for (auto op : ops)
    {
        reduce<T>(op.first, A, idx_A.data(), ref_val, ref_idx);

        reduce_init(op.first, blas_val, blas_idx);

        switch (op.first)
        {
            case REDUCE_MIN:
            case REDUCE_MIN_ABS:
                blas_val = -blas_val;
                break;
        }

        for (stride_type i = 0;i < NA;i++)
        {
            auto val = data[i];

            switch (op.first)
            {
                case REDUCE_SUM_ABS:
                case REDUCE_MAX_ABS:
                case REDUCE_MIN_ABS:
                    val = std::abs(val);
                    break;
                case REDUCE_NORM_2:
                    val = norm2(val);
                    break;
            }

            switch (op.first)
            {
                case REDUCE_MIN:
                case REDUCE_MIN_ABS:
                    val = -val;
                    break;
            }

            switch (op.first)
            {
                case REDUCE_SUM:
                case REDUCE_SUM_ABS:
                case REDUCE_NORM_2:
                    blas_val += val;
                    break;
                case REDUCE_MAX:
                case REDUCE_MAX_ABS:
                case REDUCE_MIN:
                case REDUCE_MIN_ABS:
                    if (val > blas_val)
                    {
                        blas_val = val;
                        blas_idx = i;
                    }
                    break;
            }
        }

        switch (op.first)
        {
            case REDUCE_MIN:
            case REDUCE_MIN_ABS:
                blas_val = -blas_val;
                break;
            case REDUCE_NORM_2:
                blas_val = sqrt(blas_val);
                break;
        }

        passfail(op.second, ref_idx, blas_idx, ref_val, blas_val, ulp_factor*ceil2(NA));
    }

    A = T(1);
    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    passfail("COUNT", ref_val, NA, ulp_factor*ceil2(NA));
}

template <typename T>
void test(stride_type N_in_bytes, int R)
{
    stride_type N = N_in_bytes/sizeof(T);

    //for (int i = 0;i < R;i++) test_tblis<T>(N);

    //for (int i = 0;i < R;i++) test_reduce<T>(N);
    //for (int i = 0;i < R;i++) test_scale<T>(N);
    //for (int i = 0;i < R;i++) test_transpose<T>(N);
    //for (int i = 0;i < R;i++) test_dot<T>(N);
    //for (int i = 0;i < R;i++) test_replicate<T>(N);
    //for (int i = 0;i < R;i++) test_trace<T>(N);
    //for (int i = 0;i < R;i++) test_add<T>(N);
    //for (int i = 0;i < R;i++) test_outer_prod<T>(N);
    //for (int i = 0;i < R;i++) test_weight<T>(N);
    //for (int i = 0;i < R;i++) test_contract<T>(N);
    //for (int i = 0;i < R;i++) test_mult<T>(N);

    for (int i = 0;i < R;i++) test_dpd_reduce<T>(N);
    for (int i = 0;i < R;i++) test_dpd_scale<T>(N);
    for (int i = 0;i < R;i++) test_dpd_transpose<T>(N);
    for (int i = 0;i < R;i++) test_dpd_dot<T>(N);
    for (int i = 0;i < R;i++) test_dpd_replicate<T>(N);
    for (int i = 0;i < R;i++) test_dpd_trace<T>(N);
    for (int i = 0;i < R;i++) test_dpd_add<T>(N);
    for (int i = 0;i < R;i++) test_dpd_outer_prod<T>(N);
    for (int i = 0;i < R;i++) test_dpd_weight<T>(N);
    for (int i = 0;i < R;i++) test_dpd_contract<T>(N);
    for (int i = 0;i < R;i++) test_dpd_mult<T>(N);
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

    cout << "Running tests with " << tblis_get_num_threads() << " threads" << endl;

    test<   float>(N, R);
    test<  double>(N, R);
    test<scomplex>(N, R);
    test<dcomplex>(N, R);

    return 0;
}
