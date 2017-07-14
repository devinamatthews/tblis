#include <algorithm>
#include <limits>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <iomanip>
#include <map>
#include <typeinfo>
#include <cxxabi.h>

#include "tblis.h"

#include "internal/3t/mult.hpp"
#include "internal/3t/dpd_mult.hpp"
#include "util/random.hpp"
#include "util/tensor.hpp"
#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/iostream.hpp"

#define CATCH_CONFIG_RUNNER
#include "external/catch/catch.hpp"

using namespace std;
using namespace stl_ext;
using namespace tblis;
using namespace tblis::internal;
using namespace tblis::detail;
using namespace tblis::slice;

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

template <typename... Types> struct types;

template <template <typename> class Body, typename... Types> struct templated_test_case_runner;

template <template <typename> class Body, typename... Types>
struct templated_test_case_runner<Body, types<Types...>>
{
    static void run()
    {
        templated_test_case_runner<Body, Types...>::run();
    }
};

template <template <typename> class Body, typename Type, typename... Types>
struct templated_test_case_runner<Body, Type, Types...>
{
    static void run()
    {
        {
            INFO("Template parameter: " << type_name<Type>());
            Body<Type>::run();
        }
        templated_test_case_runner<Body, Types...>::run();
    }
};

template <template <typename> class Body>
struct templated_test_case_runner<Body>
{
    static void run() {}
};

#define REPLICATED_TEST_CASE(name, ntrial) \
static void TBLIS_PASTE(__replicated_test_case_body_, name)(); \
TEST_CASE(#name) \
{ \
    for (int trial = 0;trial < ntrial;trial++) \
    { \
        INFO("Trial " << (trial+1) << " of " << ntrial); \
        TBLIS_PASTE(__replicated_test_case_body_, name)(); \
    } \
} \
static void TBLIS_PASTE(__replicated_test_case_body_, name)()

#define TEMPLATED_TEST_CASE(name, T, ...) \
template <typename T> struct TBLIS_PASTE(__templated_test_case_body_, name) \
{ \
    static void run(); \
}; \
TEST_CASE(#name) \
{ \
    templated_test_case_runner<TBLIS_PASTE(__templated_test_case_body_, name), __VA_ARGS__>::run(); \
} \
template <typename T> void TBLIS_PASTE(__templated_test_case_body_, name)<T>::run()

#define REPLICATED_TEMPLATED_TEST_CASE(name, ntrial, T, ...) \
template <typename T> static void TBLIS_PASTE(__replicated_templated_test_case_body_, name)(); \
TEMPLATED_TEST_CASE(name, T, __VA_ARGS__) \
{ \
    for (int trial = 0;trial < ntrial;trial++) \
    { \
        INFO("Trial " << (trial+1) << " of " << ntrial); \
        TBLIS_PASTE(__replicated_templated_test_case_body_, name)<T>(); \
    } \
} \
template <typename T> static void TBLIS_PASTE(__replicated_templated_test_case_body_, name)()

constexpr static int ulp_factor = 32;

static stride_type N = 1024*1024;
static int R = 10;
typedef types<float, double, scomplex, dcomplex> all_types;

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
len_vector group_size(const matrix<len_type>& len, const T& idx, const T& choose)
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

    len_vector size(nirrep);
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

template <typename T, typename U>
void check(const string& label, stride_type ia, stride_type ib, T error, U ulps)
{
    typedef decltype(std::abs(error)) V;
    auto epsilon = std::abs(max(numeric_limits<V>::min(),
       float(ceil2(ulp_factor*std::abs(ulps)))*numeric_limits<V>::epsilon()));

    INFO(label);
    INFO("Error = " << std::abs(error));
    INFO("Epsilon = " << epsilon);
    REQUIRE(std::abs(error) == Approx(0).epsilon(0).margin(epsilon));
    REQUIRE(ia == ib);
}

template <typename T, typename U>
void check(const string& label, T error, U ulps)
{
    check(label, 0, 0, error, ulps);
}

template <typename T, typename U, typename V>
void check(const string& label, stride_type ia, stride_type ib, T a, U b, V ulps)
{
    INFO("Values = " << a << ", " << b);
    check(label, ia, ib, a-b, ulps);
}

template <typename T, typename U, typename V>
void check(const string& label, T a, U b, V ulps)
{
    check(label, 0, 0, a, b, ulps);
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
    len_vector len = random_product_constrained_sequence<len_type>(2, N/sizeof(T), {m_min, n_min});

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
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type n = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type k = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));

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
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type k = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));

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
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type n = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));

    random_matrix(N, m, 1, A);
    random_matrix(N, 1, n, B);
    random_matrix(N, m, n, C);
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

REPLICATED_TEMPLATED_TEST_CASE(gemm, R, T, all_types)
{
    matrix<T> A, B, C, D, E;

    random_gemm(N, A, B, C);

    T scale(10.0*random_unit<T>());

    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    INFO("m, n, k    = " << m << ", " << n << ", " << k);
    INFO("rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1));
    INFO("rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1));
    INFO("rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1));

    D.reset(C);
    gemm_ref<T>(scale, A, B, scale, D);

    E.reset(C);
    mult<T>(scale, A, B, scale, E);

    add<T>(T(-1), D, T(1), E);
    T error = reduce<T>(REDUCE_NORM_2, E).first;

    check("REF", error, scale*m*n*k);
}

REPLICATED_TEMPLATED_TEST_CASE(gemv, R, T, all_types)
{
    matrix<T> A, B, C, D, E;

    random_gemv(N, A, B, C);

    T scale(10.0*random_unit<T>());

    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    INFO("m, n, k    = " << m << ", " << n << ", " << k);
    INFO("rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1));
    INFO("rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1));
    INFO("rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1));

    D.reset(C);
    gemm_ref<T>(scale, A, B, scale, D);

    E.reset(C);
    mult<T>(scale, A, B, scale, E);

    add<T>(T(-1), D, T(1), E);
    T error = reduce<T>(REDUCE_NORM_2, E).first;

    check("REF", error, scale*m*n*k);
}

REPLICATED_TEMPLATED_TEST_CASE(ger, R, T, all_types)
{
    matrix<T> A, B, C, D, E;

    random_ger(N, A, B, C);

    T scale(10.0*random_unit<T>());

    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    INFO("m, n, k    = " << m << ", " << n << ", " << k);
    INFO("rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1));
    INFO("rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1));
    INFO("rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1));

    D.reset(C);
    gemm_ref<T>(scale, A, B, scale, D);

    E.reset(C);
    mult<T>(scale, A, B, scale, E);

    add<T>(T(-1), D, T(1), E);
    T error = reduce<T>(REDUCE_NORM_2, E).first;

    check("REF", error, scale*m*n*k);
}

/*
 * Creates a tensor of d dimensions, whose total storage size is between N/2^d
 * and N entries, and with edge lengths of at least those given. The number
 * of referencable elements between N/4^d and N/2^d. Non-referencable elements
 * are initialized to zero, while referencable elements are randomly
 * initialized from the interior of the unit circle.
 */
void random_lengths(stride_type N, unsigned d, const vector<len_type>& len_min, len_vector& len)
{
    len_vector len_max = random_product_constrained_sequence<len_type>(d, N, len_min);

    len.resize(d);
    for (unsigned i = 0;i < d;i++)
    {
        len[i] = (len_min[i] > 0 ? len_min[i] : random_number<len_type>(1, len_max[i]));
    }
}

matrix<len_type> random_indices(const len_vector& len, double sparsity)
{
    matrix<len_type> idx({(len_type)len.size(), prod(len)});

    len_type i = 0;
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

    return idx[all][range(max<len_type>(i,1))];
}

template <typename T>
void randomize_tensor(T& t)
{
    typedef typename T::value_type U;
    t.for_each_element([](U& e) { e = random_unit<U>(); });
}

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, varray<T>& A)
{
    len_vector len_A;
    random_lengths(N/sizeof(T), d, len_min, len_A);
    A.reset(len_A);
    randomize_tensor(A);
}

template <typename T>
void random_tensor(stride_type N, unsigned d, unsigned nirrep, const len_vector& len_min, dpd_varray<T>& A)
{
    unsigned irrep_A;
    len_vector len_A(d);

    do
    {
        irrep_A = random_number(nirrep-1);

        len_vector len_A_;
        random_lengths(nirrep*N/sizeof(T), d, len_min, len_A_);

        for (unsigned i = 0;i < d;i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);
    }
    while (A.size(irrep_A, len_A) == 0);

    A.reset(irrep_A, nirrep, len_A);
    randomize_tensor(A);
}

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, indexed_varray<T>& A)
{
    len_vector len_A;
    random_lengths(N/sizeof(T), d, len_min, len_A);

    unsigned dense_d = random_number(1u, d-1);
    auto idxs_A = random_indices(len_vector(len_A.begin()+dense_d, len_A.end()), 0.1);

    A.reset(len_A, idxs_A);
    randomize_tensor(A);
}

template <typename T>
void random_tensor(stride_type N, unsigned d, unsigned nirrep, const len_vector& len_min, indexed_dpd_varray<T>& A)
{
    unsigned irrep_A;
    len_vector len_A(d);

    do
    {
        irrep_A = random_number(nirrep-1);

        len_vector len_A_;
        random_lengths(nirrep*N/sizeof(T), d, len_min, len_A_);

        for (unsigned i = 0;i < d;i++)
            len_A[i] = random_sum_constrained_sequence<len_type>(nirrep, len_A_[i]);
    }
    while (A.size(irrep_A, len_A) == 0);

    unsigned dense_d = random_number(1u, d-1);
    len_vector idx_len_A(d-dense_d);
    irrep_vector idx_irrep_A(d-dense_d);
    for (unsigned i = dense_d;i < d;i++)
    {
        idx_irrep_A[i-dense_d] = random_number(nirrep-1);
        idx_len_A[i-dense_d] = len_A[i][idx_irrep_A[i-dense_d]];
    }
    auto idxs_A = random_indices(idx_len_A, 0.1);

    A.reset(irrep_A, nirrep, len_A, idx_irrep_A, idxs_A);
    randomize_tensor(A);
}

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, dpd_varray<T>& A)
{
    random_tensor(N, d, 1 << random_number(2), len_min, A);
}

template <typename T>
void random_tensor(stride_type N, unsigned d, const len_vector& len_min, indexed_dpd_varray<T>& A)
{
    random_tensor(N, d, 1 << random_number(2), len_min, A);
}

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

template <typename T>
void random_tensor(stride_type N, unsigned d, T& A)
{
    random_tensor(N, d, len_vector(d), A);
}

/*
 * Creates a random tensor of 1 to 8 dimensions.
 */
void random_lengths(stride_type N, len_vector& len)
{
    random_lengths(N, random_number(1,8), len);
}

template <typename T>
void random_tensor(stride_type N, T& A)
{
    random_tensor(N, random_number(1,8), A);
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
    label_vector idx_A_only(ndim_A_only);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    label_vector idx_B_only(ndim_B_only);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    label_vector idx_AB(ndim_AB);
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

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    dpd_varray<T>& A, label_vector& idx_A,
                    dpd_varray<T>& B, label_vector& idx_B)
{
    unsigned nirrep;
    unsigned irrep_A, irrep_B;
    len_vector len_A, len_B;

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
    while (A.size(irrep_A, len_A) == 0 ||
           B.size(irrep_B, len_B) == 0);

    A.reset(irrep_A, nirrep, len_A);
    B.reset(irrep_B, nirrep, len_B);
    randomize_tensor(A);
    randomize_tensor(B);
}

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

    unsigned dense_ndim_A = random_number(1u, ndim_AB+ndim_A_only-1);
    unsigned dense_ndim_B = random_number(1u, ndim_AB+ndim_B_only-1);

    auto idxs_A = random_indices(len_vector(len_A.begin()+dense_ndim_A, len_A.end()), 0.1);
    auto idxs_B = random_indices(len_vector(len_B.begin()+dense_ndim_B, len_B.end()), 0.1);

    A.reset(len_A, idxs_A);
    B.reset(len_B, idxs_B);

    randomize_tensor(A);
    randomize_tensor(B);
}

template <typename T>
void random_tensors(stride_type N,
                    unsigned ndim_A_only, unsigned ndim_B_only, unsigned ndim_AB,
                    indexed_dpd_varray<T>& A, label_vector& idx_A,
                    indexed_dpd_varray<T>& B, label_vector& idx_B)
{
    unsigned nirrep;
    unsigned irrep_A, irrep_B;
    len_vector len_A, len_B;

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
    while (A.size(irrep_A, len_A) == 0 ||
           B.size(irrep_B, len_B) == 0);

    unsigned ndim_A = ndim_AB+ndim_A_only;
    unsigned dense_ndim_A = random_number(1u, ndim_A-1);
    len_vector idx_len_A(ndim_A-dense_ndim_A);
    irrep_vector idx_irrep_A(ndim_A-dense_ndim_A);
    for (unsigned i = dense_ndim_A;i < ndim_A;i++)
    {
        idx_irrep_A[i-dense_ndim_A] = random_number(nirrep-1);
        idx_len_A[i-dense_ndim_A] = len_A[i][idx_irrep_A[i-dense_ndim_A]];
    }
    auto idxs_A = random_indices(idx_len_A, 0.1);

    unsigned ndim_B = ndim_AB+ndim_B_only;
    unsigned dense_ndim_B = random_number(1u, ndim_B-1);
    len_vector idx_len_B(ndim_B-dense_ndim_B);
    irrep_vector idx_irrep_B(ndim_B-dense_ndim_B);
    for (unsigned i = dense_ndim_B;i < ndim_B;i++)
    {
        idx_irrep_B[i-dense_ndim_B] = random_number(nirrep-1);
        idx_len_B[i-dense_ndim_B] = len_B[i][idx_irrep_B[i-dense_ndim_B]];
    }
    auto idxs_B = random_indices(idx_len_B, 0.1);

    A.reset(irrep_A, nirrep, len_A, idx_irrep_A, idxs_A);
    B.reset(irrep_B, nirrep, len_B, idx_irrep_B, idxs_B);
    randomize_tensor(A);
    randomize_tensor(B);
}

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
    label_vector idx_A_only(ndim_A_only);
    for (unsigned i = 0;i < ndim_A_only;i++) idx_A_only[i] = idx[c++];

    label_vector idx_B_only(ndim_B_only);
    for (unsigned i = 0;i < ndim_B_only;i++) idx_B_only[i] = idx[c++];

    label_vector idx_C_only(ndim_C_only);
    for (unsigned i = 0;i < ndim_C_only;i++) idx_C_only[i] = idx[c++];

    label_vector idx_AB(ndim_AB);
    for (unsigned i = 0;i < ndim_AB;i++) idx_AB[i] = idx[c++];

    label_vector idx_AC(ndim_AC);
    for (unsigned i = 0;i < ndim_AC;i++) idx_AC[i] = idx[c++];

    label_vector idx_BC(ndim_BC);
    for (unsigned i = 0;i < ndim_BC;i++) idx_BC[i] = idx[c++];

    label_vector idx_ABC(ndim_ABC);
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
    len_vector len_A, len_B, len_C;

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
    while (A.size(irrep_A, len_A) == 0 ||
           B.size(irrep_B, len_B) == 0 ||
           C.size(irrep_C, len_C) == 0);

    A.reset(irrep_A, nirrep, len_A);
    B.reset(irrep_B, nirrep, len_B);
    C.reset(irrep_C, nirrep, len_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
}

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

    unsigned dense_ndim_A = random_number(1u, ndim_ABC+ndim_AB+ndim_AC+ndim_A_only-1);
    unsigned dense_ndim_B = random_number(1u, ndim_ABC+ndim_AB+ndim_BC+ndim_B_only-1);
    unsigned dense_ndim_C = random_number(1u, ndim_ABC+ndim_AC+ndim_BC+ndim_C_only-1);

    auto idxs_A = random_indices(len_vector(len_A.begin()+dense_ndim_A, len_A.end()), 0.1);
    auto idxs_B = random_indices(len_vector(len_B.begin()+dense_ndim_B, len_B.end()), 0.1);
    auto idxs_C = random_indices(len_vector(len_C.begin()+dense_ndim_C, len_C.end()), 0.1);

    A.reset(len_A, idxs_A);
    B.reset(len_B, idxs_B);
    C.reset(len_C, idxs_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
}

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
    len_vector len_A, len_B, len_C;

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
    while (A.size(irrep_A, len_A) == 0 ||
           B.size(irrep_B, len_B) == 0 ||
           C.size(irrep_C, len_C) == 0);

    unsigned ndim_A = ndim_ABC+ndim_AB+ndim_AC+ndim_A_only;
    unsigned dense_ndim_A = random_number(1u, ndim_A-1);
    len_vector idx_len_A(ndim_A-dense_ndim_A);
    irrep_vector idx_irrep_A(ndim_A-dense_ndim_A);
    for (unsigned i = dense_ndim_A;i < ndim_A;i++)
    {
        idx_irrep_A[i-dense_ndim_A] = random_number(nirrep-1);
        idx_len_A[i-dense_ndim_A] = len_A[i][idx_irrep_A[i-dense_ndim_A]];
    }
    auto idxs_A = random_indices(idx_len_A, 0.1);

    unsigned ndim_B = ndim_ABC+ndim_AB+ndim_BC+ndim_B_only;
    unsigned dense_ndim_B = random_number(1u, ndim_B-1);
    len_vector idx_len_B(ndim_B-dense_ndim_B);
    irrep_vector idx_irrep_B(ndim_B-dense_ndim_B);
    for (unsigned i = dense_ndim_B;i < ndim_B;i++)
    {
        idx_irrep_B[i-dense_ndim_B] = random_number(nirrep-1);
        idx_len_B[i-dense_ndim_B] = len_B[i][idx_irrep_B[i-dense_ndim_B]];
    }
    auto idxs_B = random_indices(idx_len_B, 0.1);

    unsigned ndim_C = ndim_ABC+ndim_AC+ndim_BC+ndim_C_only;
    unsigned dense_ndim_C = random_number(1u, ndim_C-1);
    len_vector idx_len_C(ndim_C-dense_ndim_C);
    irrep_vector idx_irrep_C(ndim_C-dense_ndim_C);
    for (unsigned i = dense_ndim_C;i < ndim_C;i++)
    {
        idx_irrep_C[i-dense_ndim_C] = random_number(nirrep-1);
        idx_len_C[i-dense_ndim_C] = len_C[i][idx_irrep_C[i-dense_ndim_C]];
    }
    auto idxs_C = random_indices(idx_len_C, 0.1);

    A.reset(irrep_A, nirrep, len_A, idx_irrep_A, idxs_A);
    B.reset(irrep_B, nirrep, len_B, idx_irrep_B, idxs_B);
    C.reset(irrep_C, nirrep, len_C, idx_irrep_C, idxs_C);

    randomize_tensor(A);
    randomize_tensor(B);
    randomize_tensor(C);
}

static map<reduce_t, string> ops =
{
 {REDUCE_SUM, "REDUCE_SUM"},
 {REDUCE_SUM_ABS, "REDUCE_SUM_ABS"},
 {REDUCE_MAX, "REDUCE_MAX"},
 {REDUCE_MAX_ABS, "REDUCE_MAX_ABS"},
 {REDUCE_MIN, "REDUCE_MIN"},
 {REDUCE_MIN_ABS, "REDUCE_MIN_ABS"},
 {REDUCE_NORM_2, "REDUCE_NORM_2"}
};

REPLICATED_TEMPLATED_TEST_CASE(reduce, R, T, all_types)
{
    varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INFO("len    = " << A.lengths());
    INFO("stride = " << A.strides());

    auto NA = prod(A.lengths());

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

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

        check(op.second, ref_idx, blas_idx, ref_val, blas_val, NA);
    }

    A = T(1);
    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    check("COUNT", ref_val, NA, NA);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_reduce, R, T, all_types)
{
    dpd_varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INFO("irrep = " << A.irrep());
    INFO("len   = \n" << A.lengths());

    auto NA = A.size(A.irrep(), A.lengths());

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

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

        check(op.second, ref_idx, blas_idx, ref_val, blas_val, NA);
    }

    A = T(1);
    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    check("COUNT", ref_val, NA, NA);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_reduce, R, T, all_types)
{
    indexed_varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INFO("dense len    = " << A.dense_lengths());
    INFO("dense stride = " << A.dense_strides());
    INFO("indexed len  = " << A.indexed_lengths());

    auto NA = prod(A.dense_lengths())*A.num_indices();

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

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

        check(op.second, ref_idx, blas_idx, ref_val, blas_val, NA);
    }

    A = T(1);
    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    check("COUNT", ref_val, NA, NA);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_reduce, R, T, all_types)
{
    indexed_dpd_varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INFO("irrep          = " << A.irrep());
    INFO("dense len      = \n" << A.dense_lengths());
    INFO("indexed irreps = " << A.indexed_irreps());
    INFO("indexed len    = \n" << A.indexed_lengths());

    auto NA = A.size(A.dense_irrep(), A.dense_lengths())*A.num_indices();

    T ref_val, blas_val;
    stride_type ref_idx, blas_idx;

    T* data = A.data();

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

        check(op.second, ref_idx, blas_idx, ref_val, blas_val, NA);
    }

    A = T(1);
    reduce<T>(REDUCE_SUM, A, idx_A.data(), ref_val, ref_idx);
    check("COUNT", ref_val, NA, NA);
}

REPLICATED_TEMPLATED_TEST_CASE(scale, R, T, all_types)
{
    varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INFO("len    = " << A.lengths());
    INFO("stride = " << A.strides());

    auto neps = prod(A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    tblis::scale<T>(scale, A, idx_A.data());
    T calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("RANDOM", ref_val, calc_val/scale, neps);

    tblis::scale<T>(T(1), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("UNIT", ref_val, calc_val/scale, neps);

    tblis::scale<T>(T(0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("ZERO", calc_val, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_scale, R, T, all_types)
{
    dpd_varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INFO("irrep = " << A.irrep());
    INFO("len   = \n" << A.lengths());

    auto neps = A.size(A.irrep(), A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    tblis::scale<T>(scale, A, idx_A.data());
    T calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("RANDOM", ref_val, calc_val/scale, neps);

    tblis::scale<T>(T(1.0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("UNIT", ref_val, calc_val/scale, neps);

    tblis::scale<T>(T(0.0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("ZERO", calc_val, neps);
}

/*
 * Creates a random tensor trace operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_trace(stride_type N, T&& A, label_vector& idx_A,
                                 T&& B, label_vector& idx_B)
{
    unsigned ndim_A, ndim_B;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        if (ndim_A < ndim_B) swap(ndim_A, ndim_B);
    }
    while (ndim_A == ndim_B);

    random_tensors(N,
                   ndim_A-ndim_B, 0,
                   ndim_B,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(trace, R, T, all_types)
{
    varray<T> A, B;
    label_vector idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    INFO("len_A    = " << A.lengths());
    INFO("stride_A = " << A.strides());
    INFO("idx_A    = " << idx_A);
    INFO("len_B    = " << B.lengths());
    INFO("stride_B = " << B.strides());
    INFO("idx_B    = " << idx_B);

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    check("SUM", scale*(ref_val+add_b), calc_val, neps*scale);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_trace, R, T, all_types)
{
    dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    INFO("irrep_A = " << A.irrep());
    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);

    auto neps = A.size(A.irrep(), A.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = FULL;
    C.reset(B);
    add<T>(scale, A, idx_A.data(), scale, C, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A.data(), scale, D, idx_B.data());

    add<T>(T(-1), C, idx_B.data(), T(1), D, idx_B.data());
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B.data()).first;

    check("BLOCKED", error, scale*neps);
}

/*
 * Creates a random tensor replication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_replicate(stride_type N, T&& A, label_vector& idx_A,
                                     T&& B, label_vector& idx_B)
{
    unsigned ndim_A, ndim_B;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        if (ndim_B < ndim_A) swap(ndim_A, ndim_B);
    }
    while (ndim_A == ndim_B);

    random_tensors(N,
                   0, ndim_B-ndim_A,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(replicate, R, T, all_types)
{
    varray<T> A, B;
    label_vector idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

    INFO("len_A    = " << A.lengths());
    INFO("stride_A = " << A.strides());
    INFO("idx_A    = " << idx_A);
    INFO("len_B    = " << B.lengths());
    INFO("stride_B = " << B.strides());
    INFO("idx_B    = " << idx_B);

    auto idx_B_only = exclusion(idx_B, idx_A);
    stride_type NB = prod(select_from(B.lengths(), idx_B, idx_B_only));
    auto neps = prod(B.lengths());

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    add<T>(scale, A, idx_A.data(), scale, B, idx_B.data());
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B.data()).first;
    check("SUM", scale*(NB*ref_val+add_b), calc_val, neps*scale);

    ref_val = reduce<T>(REDUCE_NORM_1, A, idx_A.data()).first;
    add<T>(scale, A, idx_A.data(), T(0.0), B, idx_B.data());
    calc_val = reduce<T>(REDUCE_NORM_1, B, idx_B.data()).first;
    check("NRM1", std::abs(scale)*NB*ref_val, calc_val, neps*scale);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_replicate, R, T, all_types)
{
    dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("irrep_B = " << B.irrep());
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);

    auto neps = B.size(B.irrep(), B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = FULL;
    C.reset(B);
    add<T>(scale, A, idx_A.data(), scale, C, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A.data(), scale, D, idx_B.data());

    add<T>(T(-1), C, idx_B.data(), T(1), D, idx_B.data());
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B.data()).first;

    check("BLOCKED", error, scale*neps);
}

/*
 * Creates a random tensor transpose operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_transpose(stride_type N, T&& A, label_vector& idx_A,
                                     T&& B, label_vector& idx_B)
{
    unsigned ndim_A = random_number(1,8);

    random_tensors(N,
                   0, 0,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(transpose, R, T, all_types)
{
    varray<T> A, B, C;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    unsigned ndim = A.dimension();
    auto perm = relative_permutation(idx_A, idx_B);

    INFO("len    = " << A.lengths());
    INFO("stride = " << A.strides());
    INFO("perm   = " << perm);

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    C.reset(A);
    add<T>(T(1), A, idx_A.data(), T(0), B, idx_B.data());
    add<T>(scale, B, idx_B.data(), scale, C, idx_A.data());

    add<T>(-2*scale, A, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    check("INVERSE", error, 2*scale*neps);

    B.reset(A);
    idx_B = idx_A;
    label_vector idx_C(ndim);
    len_vector len_C(ndim);
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
    check("CYCLE", error, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_transpose, R, T, all_types)
{
    dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("irrep_B = " << B.irrep());
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);

    auto neps = B.size(B.irrep(), B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = FULL;
    C.reset(B);
    add<T>(scale, A, idx_A.data(), scale, C, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A.data(), scale, D, idx_B.data());

    add<T>(T(-1), C, idx_B.data(), T(1), D, idx_B.data());
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B.data()).first;

    check("BLOCKED", error, scale*neps);
}

/*
 * Creates a random tensor dot product operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_dot(stride_type N, T&& A, label_vector& idx_A,
                               T&& B, label_vector& idx_B)
{
    unsigned ndim_A = random_number(1,8);

    random_tensors(N,
                   0, 0,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(dot, R, T, all_types)
{
    varray<T> A, B;
    label_vector idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    INFO("len    = " << A.lengths());
    INFO("stride = " << A.strides());

    auto neps = prod(A.lengths());

    add<T>(T(1.0), A, idx_A.data(), T(0.0), B, idx_B.data());
    B.for_each_element([](T& e) { e = tblis::conj(e); });
    T ref_val = reduce<T>(REDUCE_NORM_2, A, idx_A.data()).first;
    T calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    check("NRM2", ref_val*ref_val, calc_val, neps);

    B = T(1);
    ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    check("UNIT", ref_val, calc_val, neps);

    B = T(0);
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());
    check("ZERO", calc_val, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_dot, R, T, all_types)
{
    dpd_varray<T> A, B, C;
    label_vector idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    INFO("irrep = " << A.irrep());
    INFO("len   = \n" << A.lengths());

    auto neps = A.size(A.irrep(), A.lengths());

    C.reset(B);
    add<T>(T(1.0), A, idx_A.data(), T(0.0), C, idx_B.data());
    C.for_each_element([](T& e) { e = tblis::conj(e); });
    T ref_val = reduce<T>(REDUCE_NORM_2, A, idx_A.data()).first;
    T calc_val = dot<T>(A, idx_A.data(), C, idx_B.data());
    check("NRM2", ref_val*ref_val, calc_val, neps);

    C = T(1);
    ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    calc_val = dot<T>(A, idx_A.data(), C, idx_B.data());
    check("UNIT", ref_val, calc_val, neps);

    C = T(0);
    calc_val = dot<T>(A, idx_A.data(), C, idx_B.data());
    check("ZERO", calc_val, neps);

    dpd_impl = FULL;
    ref_val = dot<T>(A, idx_A.data(), B, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    calc_val = dot<T>(A, idx_A.data(), B, idx_B.data());

    check("BLOCKED", calc_val, ref_val, neps);
}

/*
 * Creates a random tensor multiplication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_mult(stride_type N, T&& A, label_vector& idx_A,
                                T&& B, label_vector& idx_B,
                                T&& C, label_vector& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AB, ndim_AC, ndim_BC;
    int ndim_ABC;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = random_number(1,8);
        ndim_ABC = random_number(min({ndim_A, ndim_B, ndim_C}));
        ndim_AB  = (ndim_A+ndim_B-ndim_C-ndim_ABC)/2;
        ndim_AC = ndim_A-ndim_ABC-ndim_AB;
        ndim_BC = ndim_B-ndim_ABC-ndim_AB;
    }
    while (ndim_AB < 0 ||
           ndim_AC < 0 ||
           ndim_BC < 0 ||
           (ndim_A+ndim_B-ndim_C-ndim_ABC)%2 != 0);

    random_tensors(N,
                   0, 0, 0,
                   ndim_AB, ndim_AC, ndim_BC,
                   ndim_ABC,
                   A, idx_A,
                   B, idx_B,
                   C, idx_C);
}

REPLICATED_TEMPLATED_TEST_CASE(mult, R, T, all_types)
{
    varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_mult(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("len_A    = " << A.lengths());
    INFO("stride_A = " << A.strides());
    INFO("idx_A    = " << idx_A);
    INFO("len_B    = " << B.lengths());
    INFO("stride_B = " << B.strides());
    INFO("idx_B    = " << idx_B);
    INFO("len_C    = " << C.lengths());
    INFO("stride_C = " << C.strides());
    INFO("idx_C    = " << idx_C);

    auto idx_AB = exclusion(intersection(idx_A, idx_B), idx_C);
    auto idx_A_only = exclusion(idx_A, idx_B, idx_C);
    auto idx_B_only = exclusion(idx_B, idx_A, idx_C);

    auto neps = prod(select_from(A.lengths(), idx_A, idx_AB))*prod(C.lengths()));

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_mult, R, T, all_types)
{
    dpd_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_mult(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("irrep_A = " << A.irrep());
    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("irrep_B = " << B.irrep());
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);
    INFO("irrep_C = " << C.irrep());
    INFO("len_C   = \n" << C.lengths());
    INFO("idx_C   = " << idx_C);

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
    stride_type neps = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_ABC = A.irrep()^B.irrep()^C.irrep();
        unsigned irrep_AC = A.irrep()^irrep_AB^irrep_ABC;
        unsigned irrep_BC = B.irrep()^irrep_AB^irrep_ABC;

        neps += size_ABC[irrep_ABC]*
                size_AB[irrep_AB]*
                size_AC[irrep_AC]*
                size_BC[irrep_BC];
    }

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_contract(stride_type N, T&& A, label_vector& idx_A,
                                    T&& B, label_vector& idx_B,
                                    T&& C, label_vector& idx_C)
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

REPLICATED_TEMPLATED_TEST_CASE(contract, R, T, all_types)
{
    varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    T scale(10.0*random_unit<T>());

    INFO("len_A    = " << A.lengths());
    INFO("stride_A = " << A.strides());
    INFO("idx_A    = " << idx_A);
    INFO("len_B    = " << B.lengths());
    INFO("stride_B = " << B.strides());
    INFO("idx_B    = " << idx_B);
    INFO("len_C    = " << C.lengths());
    INFO("stride_C = " << C.strides());
    INFO("idx_C    = " << idx_C);

    auto idx_AB = intersection(idx_A, idx_B);

    auto neps = prod(select_from(A.lengths(), idx_A, idx_AB))*prod(C.lengths());

    impl = REFERENCE;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = BLAS_BASED;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);

    impl = BLIS_BASED;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLIS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_contract, R, T, all_types)
{
    dpd_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("irrep_A = " << A.irrep());
    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("irrep_B = " << B.irrep());
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);
    INFO("irrep_C = " << C.irrep());
    INFO("len_C   = \n" << C.lengths());
    INFO("idx_C   = " << idx_C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_AC = intersection(idx_A, idx_C);
    auto idx_BC = intersection(idx_B, idx_C);

    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);

    unsigned nirrep = A.num_irreps();
    stride_type neps = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_AC = A.irrep()^irrep_AB;
        unsigned irrep_BC = B.irrep()^irrep_AB;

        neps += size_AB[irrep_AB]*
                size_AC[irrep_AC]*
                size_BC[irrep_BC];
    }

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

/*
 * Creates a random tensor weighting operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_weight(stride_type N, T&& A, label_vector& idx_A,
                                  T&& B, label_vector& idx_B,
                                  T&& C, label_vector& idx_C)
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

REPLICATED_TEMPLATED_TEST_CASE(weight, R, T, all_types)
{
    varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("len_A    = " << A.lengths());
    INFO("stride_A = " << A.strides());
    INFO("idx_A    = " << idx_A);
    INFO("len_B    = " << B.lengths());
    INFO("stride_B = " << B.strides());
    INFO("idx_B    = " << idx_B);
    INFO("len_C    = " << C.lengths());
    INFO("stride_C = " << C.strides());
    INFO("idx_C    = " << idx_C);

    auto neps = prod(C.lengths());

    T scale(10.0*random_unit<T>());

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_weight, R, T, all_types)
{
    dpd_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("irrep_A = " << A.irrep());
    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("irrep_B = " << B.irrep());
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);
    INFO("irrep_C = " << C.irrep());
    INFO("len_C   = \n" << C.lengths());
    INFO("idx_C   = " << idx_C);

    auto idx_ABC = intersection(idx_A, idx_B, idx_C);
    auto idx_AC = exclusion(intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = exclusion(intersection(idx_B, idx_C), idx_ABC);

    auto size_ABC = group_size(A.lengths(), idx_A, idx_ABC);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);

    unsigned nirrep = A.num_irreps();
    unsigned irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    unsigned irrep_AC = A.irrep()^irrep_ABC;
    unsigned irrep_BC = B.irrep()^irrep_ABC;

    stride_type neps = size_ABC[irrep_ABC]*
                       size_AC[irrep_AC]*
                       size_BC[irrep_BC];

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

/*
 * Creates a random tensor outer product operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_outer_prod(stride_type N, T&& A, label_vector& idx_A,
                                      T&& B, label_vector& idx_B,
                                      T&& C, label_vector& idx_C)
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

REPLICATED_TEMPLATED_TEST_CASE(outer_prod, R, T, all_types)
{
    varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("len_A    = " << A.lengths());
    INFO("stride_A = " << A.strides());
    INFO("idx_A    = " << idx_A);
    INFO("len_B    = " << B.lengths());
    INFO("stride_B = " << B.strides());
    INFO("idx_B    = " << idx_B);
    INFO("len_C    = " << C.lengths());
    INFO("stride_C = " << C.strides());
    INFO("idx_C    = " << idx_C);

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

    check("BLAS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE("dpd outer prod", R, T, all_types)
{
    dpd_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_outer_prod(N, A, idx_A, B, idx_B, C, idx_C);

    INFO("irrep_A = " << A.irrep());
    INFO("len_A   = \n" << A.lengths());
    INFO("idx_A   = " << idx_A);
    INFO("irrep_B = " << B.irrep());
    INFO("len_B   = \n" << B.lengths());
    INFO("idx_B   = " << idx_B);
    INFO("irrep_C = " << C.irrep());
    INFO("len_C   = \n" << C.lengths());
    INFO("idx_C   = " << idx_C);

    auto neps = C.size(C.irrep(), C.lengths());

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);
}

int main(int argc, char **argv)
{
    time_t seed = time(NULL);

    struct option opts[] = {{"size", required_argument, NULL, 'n'},
                            {"rep",  required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {0, 0, 0, 0}};

    int arg;
    int index;
    while ((arg = getopt_long(argc, argv, "n:r:s:", opts, &index)) != -1)
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
            case '?':
                ::abort();
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    rand_engine.seed(seed);

    cout << "Running tests with " << tblis_get_num_threads() << " threads\n";
    cout << endl;

    const char* catch_argv[] = {"tblis::test", "-d", "yes"};

    int nfailed = Catch::Session().run(sizeof(catch_argv)/sizeof(catch_argv[0]), catch_argv);

    return nfailed ? 1 : 0;
}
