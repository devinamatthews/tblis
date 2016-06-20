#ifndef _TBLIS_UTIL_HPP_
#define _TBLIS_UTIL_HPP_

#include <vector>
#include <string>
#include <algorithm>
#include <ostream>
#include <cstdio>
#include <cstdarg>
#include <random>

#define TBLIS_STRINGIZE(...) #__VA_ARGS__
#define TBLIS_CONCAT(x,y) x##y

#ifdef DEBUG
inline void abort_with_message(const char* cond, const char* fmt, ...)
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

#define ASSERT(x,...) \
if (x) {} \
else \
{ \
    abort_with_message(#x, "" __VA_ARGS__) ; \
}
#else
#define ASSERT(x,...)
#endif

#define DEFINE_INSTANTIATIONS() \
INSTANTIATION(   float,MR,NR,KR,MR<   float>::def,NR<   float>::def,KR<   float>::def) \
INSTANTIATION(  double,MR,NR,KR,MR<  double>::def,NR<  double>::def,KR<  double>::def) \
INSTANTIATION(scomplex,MR,NR,KR,MR<scomplex>::def,NR<scomplex>::def,KR<scomplex>::def) \
INSTANTIATION(dcomplex,MR,NR,KR,MR<dcomplex>::def,NR<dcomplex>::def,KR<dcomplex>::def)

template<typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    if (!v.empty()) os << v[0];
    for (int i = 1;i < v.size();i++) os << ", " << v[i];
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

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

template <typename T> struct real_type                  { typedef T type; };
template <typename T> struct real_type<std::complex<T>> { typedef T type; };
template <typename T> using real_type_t = typename real_type<T>::type;

template <typename T> struct complex_type                  { typedef std::complex<T> type; };
template <typename T> struct complex_type<std::complex<T>> { typedef std::complex<T> type; };
template <typename T> using complex_type_t = typename complex_type<T>::type;

template <typename T> struct is_real                  { static const bool value =  true; };
template <typename T> struct is_real<std::complex<T>> { static const bool value = false; };

template <typename T> struct is_complex                  { static const bool value = false; };
template <typename T> struct is_complex<std::complex<T>> { static const bool value =  true; };

namespace util
{

constexpr inline dim_t remainder(dim_t N, dim_t B)
{
    return (B-1)-(N+B-1)%B;
}

constexpr inline dim_t round_up(dim_t N, dim_t B)
{
    return N + remainder(N, B);
}

template <typename T, typename U>
U* convert_and_align(T* x)
{
    intptr_t off = ((intptr_t)x)%alignof(U);
    return (U*)((char*)x + (off == 0 ? 0 : alignof(U)-off));
}

template <typename T, typename U>
constexpr dim_t size_as_type(dim_t n)
{
    return (n*sizeof(T) + alignof(T)-1 + sizeof(U)-1)/sizeof(U);
}

inline const char* ptr(const std::string& x)
{
    return x.c_str();
}

template <typename T>
const T* ptr(const std::vector<T>& x)
{
    return &x[0];
}

template <typename T>
const T* ptr(const T* x)
{
    return x;
}

template <typename T>
T* ptr(std::vector<T>& x)
{
    return &x[0];
}

template <typename T>
T* ptr(T* x)
{
    return x;
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
    ASSERT(n > 0);

    double s = 0;
    for (int i = 0;i < n;i++)
    {
        ASSERT(w[i] >= 0);
        s += w[i];
    }

    double c = s*RandomNumber<double>();
    for (int i = 0;i < n;i++)
    {
        if (c < w[i]) return i;
        c -= w[i];
    }

    ASSERT(0);
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
    ASSERT(n > 0);

    T s = 0;
    for (int i = 0;i < n;i++)
    {
        ASSERT(w[i] >= 0);
        s += w[i];
    }

    T c = RandomInteger(s-1);
    for (int i = 0;i < n;i++)
    {
        if (c < w[i]) return i;
        c -= w[i];
    }

    ASSERT(0);
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
    ASSERT(n > 0);
    ASSERT(s >= 0);
    ASSERT(mn.size() == n);
    ASSERT(mn[0] >= 0);

    s -= mn[0];
    ASSERT(s >= 0);

    std::vector<double> p(n+1);

    p[0] = 0;
    p[n] = 1;
    for (int i = 1;i < n;i++)
    {
        ASSERT(mn[i] >= 0);
        s -= mn[i];
        ASSERT(s >= 0);
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
    ASSERT(n > 0);
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
    ASSERT(n >  0);
    ASSERT(s >= 0);
    ASSERT(mn.size() == n);

    for (int i = 0;i < n;i++)
    {
        ASSERT(mn[i] >= 0);
        s -= mn[i];
        ASSERT(s >= 0);
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
    ASSERT(n > 0);
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
    ASSERT(n >  0);
    ASSERT(p >= 1);
    ASSERT(mn.size() == n);

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
    ASSERT(n > 0);
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
    ASSERT(n >  0);
    ASSERT(p >= 1);
    ASSERT(mn.size() == n);

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
    ASSERT(n > 0);
    return RandomProductConstrainedSequence<T,mode>(n, p, std::vector<T>(n, T(1)));
}

template <typename T, typename U>
T permute(const T& a, const U& p)
{
    ASSERT(a.size() == p.size());
    T ap(a);
    for (size_t i = 0;i < a.size();i++) ap[p[i]] = a[i];
    return ap;
}

}
}

#endif
