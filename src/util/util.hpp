#ifndef _TBLIS_UTIL_HPP_
#define _TBLIS_UTIL_HPP_

#include <vector>
#include <string>
#include <algorithm>
#include <ostream>
#include <cstdio>
#include <cstdarg>
#include <random>

#include "blis++/blis++.hpp"

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
namespace util
{

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

template <typename Container>
Container& sort(Container& c)
{
    std::sort(c.begin(), c.end());
    return c;
}

template <typename Container>
Container& unique(Container& c)
{
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}

template <typename Container>
Container& rotate(Container& c, typename Container::iterator::diff_type n)
{
    if (n > 0)
    {
        std::rotate(c.begin(), std::advance(c.begin(),n), c.end());
    }
    else if (n < 0)
    {
        std::rotate(c.begin(), std::advance(c.end(),n), c.end());
    }
    return c;
}

/*
 * C = A^B
 */
template <typename Container>
Container& set_intersection(const Container& a,
                            const Container& b,
                                  Container& c)
{
    c.reserve(std::min(a.size(), b.size()));

    typename Container::const_iterator a0 = a.begin();
    typename Container::const_iterator a1 = a.end();
    typename Container::const_iterator b0 = b.begin();
    typename Container::const_iterator b1 = b.end();
    typename Container::iterator c0 = c.begin();
    typename Container::iterator c1 = c.end();

    while (a0 != a1 && b0 != b1)
    {
        if (*a0 < *b0)
        {
            ++a0;
        }
        else if (*b0 < *a0)
        {
            ++b0;
        }
        else
        {
            if (c0 == c1)
            {
                c.push_back(*a0);
                c0 = c1 = c.end();
            }
            else
            {
                *c0 = *a0;
                ++c0;
            }
            ++a0;
            ++b0;
        }
    }
    c.erase(c0, c1);

    return c;
}

/*
 * C = A/B
 */
/*
 * A = A^B
 */
template <typename II1, typename II2>
II1 set_intersection(II1 a0, II1 a1, II2 b0, II2 b1)
{
    II1 a2 = a0;

    while (a0 != a1 && b0 != b1)
    {
        if (*a0 < *b0)
        {
            ++a0;
        }
        else if (*b0 < *a0)
        {
            ++b0;
        }
        else
        {
            *a2 = std::move(*a0);
            ++a0;
            ++b0;
            ++a2;
        }
    }

    return a2;
}

template <typename Container>
Container& set_intersection(      Container& a,
                            const Container& b)
{
    a.erase(set_intersection(a.begin(), a.end(), b.begin(), b.end()), a.end());
    return a;
}

/*
 * A = A/B
 */
template <typename Container>
Container& set_difference(const Container& a,
                          const Container& b,
                                Container& c)
{
    c.reserve(a.size());

    typename Container::const_iterator a0 = a.begin();
    typename Container::const_iterator a1 = a.end();
    typename Container::const_iterator b0 = b.begin();
    typename Container::const_iterator b1 = b.end();
    typename Container::iterator c0 = c.begin();
    typename Container::iterator c1 = c.end();

    while (a0 != a1 && b0 != b1)
    {
        if (*a0 < *b0)
        {
            if (c0 == c1)
            {
                c.push_back(*a0);
                c0 = c1 = c.end();
            }
            else
            {
                *c0 = *a0;
                ++c0;
            }
            ++a0;
        }
        else if (*b0 < *a0)
        {
            ++b0;
        }
        else
        {
            ++a0;
            ++b0;
        }
    }

    if (c1-c0 < a1-a0)
    {
        c0 = c.erase(c0, c1);
        c.insert(c0, a0, a1);
    }
    else
    {
        c0 = std::copy(a0, a1, c0);
        c.erase(c0, c1);
    }

    return c;
}

template <typename II1, typename II2>
II1 set_difference(II1 a0, II1 a1, II2 b0, II2 b1)
{
    II1 a2 = a0;

    while (a0 != a1 && b0 != b1)
    {
        if (*a0 < *b0)
        {
            *a2 = std::move(*a0);
            ++a0;
            ++a2;
        }
        else if (*b0 < *a0)
        {
            ++b0;
        }
        else
        {
            ++a0;
            ++b0;
        }
    }

    return std::move(a0, a1, a2);
}

template <typename Container>
Container& set_difference(      Container& a,
                          const Container& b)
{
    a.erase(set_difference(a.begin(), a.end(), b.begin(), b.end()), a.end());
    return a;
}

template <typename T>
typename std::conditional<std::is_same<T,char>::value,std::string,std::vector<T>>::type
range(T to)
{
    std::vector<T> r;
    r.reserve((typename std::vector<T>::size_type)to);

    for (T i = T();i < to;++i)
    {
        r.push_back(i);
    }

    return r;
}

template <typename T>
typename std::conditional<std::is_same<T,char>::value,std::string,std::vector<T>>::type
range(T from, T to)
{
    std::vector<T> r;
    r.reserve((typename std::vector<T>::size_type)(to-from));

    for (T i = from;i < to;++i)
    {
        r.push_back(i);
    }

    return r;
}

template <> inline std::string range<char>(char to)
{
    std::string r;
    r.reserve((typename std::string::size_type)to);

    for (char i = 0;i < to;++i)
    {
        r.push_back('a'+i);
    }

    return r;
}

template <> inline std::string range<char>(char from, char to)
{
    std::string r;
    r.reserve((typename std::string::size_type)(to-from));

    for (char i = 0;i < to-from;++i)
    {
        r.push_back(from+i);
    }

    return r;
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
template <> inline blis::sComplex RandomUnit<blis::sComplex>()
{
    float r, i;
    do
    {
        r = RandomUnit<float>();
        i = RandomUnit<float>();
    }
    while (r*r+i*i >= 1);

    scomplex val;
    bli_csets(r, i, val);
    return blis::cmplx(val);
}

/*
 * Returns a psuedo-random complex number unformly distirbuted in the
 * interior of the unit circle.
 */
template <> inline blis::dComplex RandomUnit<blis::dComplex>()
{
    double r, i;
    do
    {
        r = RandomUnit<double>();
        i = RandomUnit<double>();
    }
    while (r*r+i*i >= 1);

    dcomplex val;
    bli_zsets(r, i, val);
    return blis::cmplx(val);
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
