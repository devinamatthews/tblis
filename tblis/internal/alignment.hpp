#ifndef TBLIS_ALIGNMENT_HPP
#define TBLIS_ALIGNMENT_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cmath>

#include <tblis/base/types.h>

namespace tblis
{

template <typename T>
T gcd(T a, T b)
{
    a = std::abs(a);
    b = std::abs(b);

    if (a == 0) return b;
    if (b == 0) return a;

    unsigned d = __builtin_ctzl(a|b);

    a >>= __builtin_ctzl(a);
    b >>= __builtin_ctzl(b);

    while (a != b)
    {
        if (a > b)
        {
            a = (a-b)>>1;
        }
        else
        {
            b = (b-a)>>1;
        }
    }

    return a<<d;
}

template <typename T>
T lcm(T a, T b)
{
    return a*(b/gcd(a,b));
}

template <typename T, typename U>
constexpr typename std::common_type<T,U>::type remainder(T N, U B)
{
    return (B-1)-(N+B-1)%B;
}

template <typename T, typename U>
constexpr typename std::common_type<T,U>::type round_up(T N, U B)
{
    return N + remainder(N, B);
}

template <typename T, typename U>
constexpr typename std::common_type<T,U>::type ceil_div(T N, U D)
{
    return (N >= 0 ? (N+D-1)/D : (N-D+1)/D);
}

template <typename T, typename U>
constexpr typename std::common_type<T,U>::type floor_div(T N, U D)
{
    return N/D;
}

template <typename U, typename T>
U* convert_and_align(T* x)
{
    uintptr_t off = (reinterpret_cast<uintptr_t>(x))%alignof(U);
    return reinterpret_cast<U*>(reinterpret_cast<char*>(x) + (off == 0 ? 0 : alignof(U)-off));
}

template <typename T, typename U>
constexpr size_t size_as_type(size_t n)
{
    return ceil_div(n*sizeof(T) + alignof(T), sizeof(U));
}

template <typename T>
constexpr size_t size_as_type(size_t n, type_t type)
{
    return ceil_div(n*sizeof(T) + alignof(T), type_size[type]);
}

}

#endif //TBLIS_ALIGNMENT_HPP
