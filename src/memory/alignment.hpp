#ifndef _TBLIS_ALIGNMENT_HPP_
#define _TBLIS_ALIGNMENT_HPP_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tblis
{

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

template <typename T, typename U>
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

}

#endif
