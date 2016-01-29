#ifndef _TENSOR_UTIL_UTIL_HPP_
#define _TENSOR_UTIL_UTIL_HPP_

#include <vector>
#include <string>
#include <algorithm>
#include <ostream>
#include <cstdio>
#include <cstdarg>

#if __cplusplus >= 201103l
#define MOVE(x) std::move(x)
#define MOVE3(x,y,z) std::move(x,y,z)
#else
#define MOVE(x) (x)
#define MOVE3(x,y,z) std::copy(x,y,z)
#endif

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

namespace tensor
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
            *a2 = MOVE(*a0);
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
            *a2 = MOVE(*a0);
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

    return MOVE3(a0, a1, a2);
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

}
}

#endif
