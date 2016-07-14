#ifndef _TBLIS_UTIL_HPP_
#define _TBLIS_UTIL_HPP_

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#ifdef __MACH__
#include <mach/mach_time.h>
#endif

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#include "external/marray/include/varray.hpp"
#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/vector.hpp"
#include "external/stl_ext/include/complex.hpp"

#define TBLIS_STRINGIZE_(...) #__VA_ARGS__
#define TBLIS_STRINGIZE(...) TBLIS_STRINGIZE_(__VA_ARGS__)
#define TBLIS_CONCAT_(x,y) x##y
#define TBLIS_CONCAT(x,y) TBLIS_CONCAT_(x,y)
#define TBLIS_FIRST_ARG(arg,...) arg

#ifdef TBLIS_DEBUG
inline void tblis_abort_with_message(const char* cond, const char* fmt, ...)
{
    if (strlen(fmt) == 0)
    {
        fprintf(stderr, cond);
    }
    else
    {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
    }
    fprintf(stderr, "\n");
    abort();
}

#define TBLIS_ASSERT(x,...) \
if (x) {} \
else \
{ \
    tblis_abort_with_message(TBLIS_STRINGIZE(x), "" __VA_ARGS__) ; \
}
#else
#define TBLIS_ASSERT(...)
#endif

template<typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    if (!v.empty()) os << v[0];
    for (int i = 1;i < v.size();i++) os << ", " << v[i];
    os << "]";
    return os;
}

template<typename T, size_t N> std::ostream& operator<<(std::ostream& os, const std::array<T,N>& v)
{
    os << "[";
    if (N) os << v[0];
    for (int i = 1;i < N;i++) os << ", " << v[i];
    os << "]";
    return os;
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T,U>& p)
{
    os << "{" << p.first << ", " << p.second << "}";
    return os;
}

template<typename T> bool operator!(const std::vector<T>& x)
{
    return x.empty();
}

namespace tblis
{

enum reduce_t
{
    REDUCE_SUM      = 0,
    REDUCE_SUM_ABS  = 1,
    REDUCE_MAX      = 2,
    REDUCE_MAX_ABS  = 3,
    REDUCE_MIN      = 4,
    REDUCE_MIN_ABS  = 5,
    REDUCE_NORM_1   = REDUCE_SUM_ABS,
    REDUCE_NORM_2   = 6,
    REDUCE_NORM_INF = REDUCE_MAX_ABS
};

using idx_type = MArray::varray<float>::idx_type;
using stride_type = MArray::varray<float>::stride_type;

template <typename T>
using const_tensor_view = MArray::const_varray_view<T>;

template <typename T>
using tensor_view = MArray::varray_view<T>;

template <typename T, typename Allocator=MArray::aligned_allocator<T,MARRAY_BASE_ALIGNMENT>>
using tensor = MArray::varray<T, Allocator>;

using MArray::const_marray_view;
using MArray::marray_view;
using MArray::marray;

using MArray::const_matrix_view;
using MArray::matrix_view;
using MArray::matrix;

using MArray::const_row_view;
using MArray::row_view;
using MArray::row;

using MArray::Layout;
using MArray::COLUMN_MAJOR;
using MArray::ROW_MAJOR;
using MArray::DEFAULT;

using MArray::make_array;
using MArray::make_vector;

using MArray::range_t;
using MArray::range;

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

using stl_ext::real;
using stl_ext::imag;
using stl_ext::conj;
using stl_ext::real_type_t;
using stl_ext::complex_type_t;
using stl_ext::is_complex;
using stl_ext::norm2;

inline long envtol(const std::string& env, long fallback=0)
{
    char* str = getenv(env.c_str());
    if (str) return strtol(str, nullptr, 10);
    return fallback;
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> remainder(T N, U B)
{
    return (B-1)-(N+B-1)%B;
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> round_up(T N, U B)
{
    return N + remainder(N, B);
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> ceil_div(T N, U D)
{
    return (N >= 0 ? (N+D-1)/D : (N-D+1)/D);
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> floor_div(T N, U D)
{
    return N/D;
}

template <typename T, typename U>
U* convert_and_align(T* x)
{
    intptr_t off = ((intptr_t)x)%alignof(U);
    return (U*)((char*)x + (off == 0 ? 0 : alignof(U)-off));
}

template <typename T, typename U>
constexpr size_t size_as_type(size_t n)
{
    return ceil_div(n*sizeof(T) + alignof(T), sizeof(U));
}

template <typename T>
using gemm_ukr_t =
void (*)(stride_type k,
         const T* restrict alpha,
         const T* restrict a, const T* restrict b,
         const T* restrict beta,
         T* restrict c, stride_type rs_c, stride_type cs_c,
         const void* restrict data, const void* restrict cntx);

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
}

template <typename... Args> struct has_member_helper;
template <typename... Args>
using has_member = stl_ext::conditional_t<false,
    has_member_helper<Args...>,void>;

template <typename T> struct blocksize_traits_helper;

template <typename T, size_t N>
struct blocksize_traits_helper<std::array<T,N>>
{
    static constexpr size_t value = N;
};

template <typename T>
struct blocksize_traits
{
    private:
        template<typename U>
        static std::array<int,U::max> _max_helper(U*);
        static std::array<int,T::def> _max_helper(...);

        template<typename U>
        static std::array<int,U::iota> _iota_helper(U*);
        static std::array<int, T::def> _iota_helper(...);

        template<typename U>
        static std::array<int,U::extent> _extent_helper(U*);
        static std::array<int,   T::def> _extent_helper(...);

    public:
        static constexpr idx_type def = T::def;
        static constexpr idx_type max = blocksize_traits_helper<decltype(_max_helper((T*)0))>::value;
        static constexpr idx_type iota = blocksize_traits_helper<decltype(_iota_helper((T*)0))>::value;
        static constexpr idx_type extent = blocksize_traits_helper<decltype(_extent_helper((T*)0))>::value;
};

namespace util
{

inline double tic()
{
    #ifdef __MACH__
    static double conv = -1.0;
    if (conv < 0)
    {
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        conv = (double)timebase.numer / (double)timebase.denom;
    }
    uint64_t nsec = mach_absolute_time();
    return conv*(double)nsec/1e9;
    #else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec+(double)ts.tv_nsec/1e9;
    #endif
}

extern std::mt19937 engine;

/*
 * Returns a random integer uniformly distributed in the range [mn,mx]
 */
inline
int64_t RandomInteger(int64_t mn, int64_t mx)
{
    std::uniform_int_distribution<int64_t> d(mn, mx);
    return d(engine);
}

/*
 * Returns a random integer uniformly distributed in the range [0,mx]
 */
inline
int64_t RandomInteger(int64_t mx)
{
    return RandomInteger(0, mx);
}

/*
 * Returns a pseudo-random number uniformly distributed in the range [0,1).
 */
template <typename T> T RandomNumber()
{
    std::uniform_real_distribution<T> d;
    return d(engine);
}

/*
 * Returns a pseudo-random number uniformly distributed in the range (-1,1).
 */
template <typename T> T RandomUnit()
{
    double val;
    do
    {
        val = 2*RandomNumber<T>()-1;
    } while (val == -1.0);
    return val;
}

/*
 * Returns a psuedo-random complex number unformly distirbuted in the
 * interior of the unit circle.
 */
template <> inline scomplex RandomUnit<scomplex>()
{
    float r, i;
    do
    {
        r = RandomUnit<float>();
        i = RandomUnit<float>();
    }
    while (r*r+i*i >= 1);

    return {r, i};
}

/*
 * Returns a psuedo-random complex number unformly distirbuted in the
 * interior of the unit circle.
 */
template <> inline dcomplex RandomUnit<dcomplex>()
{
    double r, i;
    do
    {
        r = RandomUnit<double>();
        i = RandomUnit<double>();
    }
    while (r*r+i*i >= 1);

    return {r, i};
}

inline
bool RandomChoice()
{
    return RandomInteger(1);
}

/*
 * Returns a random choice from a set of objects with non-negative weights w,
 * which do not need to sum to unity.
 */
inline
int RandomWeightedChoice(const std::vector<double>& w)
{
    int n = w.size();
    assert(n > 0);

    double s = 0;
    for (int i = 0;i < n;i++)
    {
        assert(w[i] >= 0);
        s += w[i];
    }

    double c = s*RandomNumber<double>();
    for (int i = 0;i < n;i++)
    {
        if (c < w[i]) return i;
        c -= w[i];
    }

    assert(0);
    return -1;
}

/*
 * Returns a random choice from a set of objects with non-negative weights w,
 * which do not need to sum to unity.
 */
template <typename T>
typename std::enable_if<std::is_integral<T>::value,int>::type
RandomWeightedChoice(const std::vector<T>& w)
{
    int n = w.size();
    assert(n > 0);

    T s = 0;
    for (int i = 0;i < n;i++)
    {
        assert(w[i] >= 0);
        s += w[i];
    }

    T c = RandomInteger(s-1);
    for (int i = 0;i < n;i++)
    {
        if (c < w[i]) return i;
        c -= w[i];
    }

    assert(0);
    return -1;
}

/*
 * Returns a sequence of n non-negative numbers such that sum_i n_i = s and
 * and n_i >= mn_i, with uniform distribution.
 */
inline
std::vector<double> RandomSumConstrainedSequence(int n, double s,
                                                 const std::vector<double>& mn)
{
    assert(n > 0);
    assert(s >= 0);
    assert(mn.size() == n);
    assert(mn[0] >= 0);

    s -= mn[0];
    assert(s >= 0);

    std::vector<double> p(n+1);

    p[0] = 0;
    p[n] = 1;
    for (int i = 1;i < n;i++)
    {
        assert(mn[i] >= 0);
        s -= mn[i];
        assert(s >= 0);
        p[i] = RandomNumber<double>();
    }
    std::sort(p.begin(), p.end());

    for (int i = 0;i < n;i++)
    {
        p[i] = s*(p[i+1]-p[i])+mn[i];
    }
    p.resize(n);
    //cout << s << p << accumulate(p.begin(), p.end(), 0.0) << endl;

    return p;
}

/*
 * Returns a sequence of n non-negative numbers such that sum_i n_i = s,
 * with uniform distribution.
 */
inline
std::vector<double> RandomSumConstrainedSequence(int n, double s)
{
    assert(n > 0);
    return RandomSumConstrainedSequence(n, s, std::vector<double>(n));
}

/*
 * Returns a sequence of n non-negative integers such that sum_i n_i = s and
 * and n_i >= mn_i, with uniform distribution.
 */
template <typename T>
typename std::enable_if<std::is_integral<T>::value,std::vector<T>>::type
RandomSumConstrainedSequence(int n, T s, const std::vector<T>& mn)
{
    assert(n >  0);
    assert(s >= 0);
    assert(mn.size() == n);

    for (int i = 0;i < n;i++)
    {
        assert(mn[i] >= 0);
        s -= mn[i];
        assert(s >= 0);
    }

    std::vector<T> p(n+1);

    p[0] = 0;
    p[n] = 1;
    for (int i = 1;i < n;i++)
    {
        p[i] = RandomInteger(s);
    }
    std::sort(p.begin(), p.end());

    for (int i = 0;i < n;i++)
    {
        p[i] = s*(p[i+1]-p[i])+mn[i];
    }
    p.resize(n);
    //cout << s << p << accumulate(p.begin(), p.end(), T(0)) << endl;

    return p;
}

/*
 * Returns a sequence of n non-negative integers such that sum_i n_i = s,
 * with uniform distribution.
 */
template <typename T>
typename std::enable_if<std::is_integral<T>::value,std::vector<T>>::type
RandomSumConstrainedSequence(int n, T s)
{
    assert(n > 0);
    return RandomSumConstrainedSequence(n, s, std::vector<T>(n));
}

/*
 * Returns a sequence of n numbers such than prod_i n_i = p and n_i >= mn_i,
 * where n_i and p are >= 1 and with uniform distribution.
 */
inline
std::vector<double> RandomProductConstrainedSequence(int n, double p,
                                                     const std::vector<double>& mn)
{
    assert(n >  0);
    assert(p >= 1);
    assert(mn.size() == n);

    std::vector<double> log_mn(n);
    for (int i = 0;i < n;i++)
    {
        log_mn[i] = (mn[i] <= 0.0 ? 1.0 : log(mn[i]));
    }

    std::vector<double> s = RandomSumConstrainedSequence(n, log(p), log_mn);
    for (int i = 0;i < n;i++) s[i] = exp(s[i]);
    //cout << p << s << accumulate(s.begin(), s.end(), 1.0, multiplies<double>()) << endl;
    return s;
}

/*
 * Returns a sequence of n numbers such than prod_i n_i = p, where n_i and
 * p are >= 1 and with uniform distribution.
 */
inline
std::vector<double> RandomProductConstrainedSequence(int n, double p)
{
    assert(n > 0);
    return RandomProductConstrainedSequence(n, p, std::vector<double>(n, 1.0));
}

enum rounding_mode {ROUND_UP, ROUND_DOWN, ROUND_NEAREST};

/*
 * Returns a sequence of n numbers such that p/2^d <= prod_i n_i <= p and
 * n_i >= mn_i, where n_i and p are >= 1 and with uniform distribution.
 */
template <typename T, rounding_mode mode=ROUND_DOWN>
typename std::enable_if<std::is_integral<T>::value,std::vector<T>>::type
RandomProductConstrainedSequence(int n, T p, const std::vector<T>& mn)
{
    assert(n >  0);
    assert(p >= 1);
    assert(mn.size() == n);

    std::vector<double> mnd(n);
    for (int i = 0;i < n;i++)
    {
        mnd[i] = std::max(T(1), mn[i]);
    }

    std::vector<double> sd = RandomProductConstrainedSequence(n, (double)p, mnd);
    std::vector<T> si(n);
    for (int i = 0;i < n;i++)
    {
        switch (mode)
        {
            case      ROUND_UP: si[i] =  ceil(sd[i]); break;
            case    ROUND_DOWN: si[i] = floor(sd[i]); break;
            case ROUND_NEAREST: si[i] = round(sd[i]); break;
        }
        si[i] = std::max(si[i], mn[i]);
    }
    //cout << p << si << accumulate(si.begin(), si.end(), T(1), multiplies<T>()) << endl;

    return si;
}

/*
 * Returns a sequence of n numbers such that p/2^d <= prod_i n_i <= p, where
 * n_i and p are >= 1 and with uniform distribution.
 */
template <typename T, rounding_mode mode=ROUND_DOWN>
typename std::enable_if<std::is_integral<T>::value,std::vector<T>>::type
RandomProductConstrainedSequence(int n, T p)
{
    assert(n > 0);
    return RandomProductConstrainedSequence<T,mode>(n, p, std::vector<T>(n, T(1)));
}

template <typename T, typename U>
T permute(const T& a, const U& p)
{
    assert(a.size() == p.size());
    T ap(a);
    for (size_t i = 0;i < a.size();i++) ap[p[i]] = a[i];
    return ap;
}

}
}

#endif
