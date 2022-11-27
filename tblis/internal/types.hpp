#ifndef TBLIS_INTERNAL_TYPES_HPP
#define TBLIS_INTERNAL_TYPES_HPP 1

#include <tblis/base/types.h>

#include <cstring>
#include <array>
#include <algorithm>
#include <type_traits>
#include <initializer_list>

#include <tblis/internal/error.hpp>

#include <marray/range.hpp>
#include <marray/short_vector.hpp>

namespace tblis
{

using MArray::short_vector;
using MArray::range;
using MArray::detail::ipow;
using MArray::detail::divide;

using len_vector = short_vector<len_type,TBLIS_OPT_NDIM>;
using stride_vector = short_vector<stride_type,TBLIS_OPT_NDIM>;
using label_vector = short_vector<label_type,TBLIS_OPT_NDIM>;
using dim_vector = short_vector<int,TBLIS_OPT_NDIM>;

inline float norm2(float x) { return x*x; }

inline double norm2(double x) { return x*x; }

inline float norm2(const scomplex& x) { return std::norm(x); }

inline double norm2(const dcomplex& x) { return std::norm(x); }

template <typename T>
bool operator<(const std::complex<T>& lhs, const std::complex<T>& rhs)
{
    return lhs.real() < rhs.real() ? true :
           lhs.real() > rhs.real() ? false :
           lhs.imag() < rhs.imag();
}

template <typename T>
bool operator<(const std::complex<T>& lhs, T rhs)
{
    return lhs.real() < rhs ? true :
           lhs.real() > rhs ? false :
           lhs.imag() < T();
}

template <typename T>
bool operator<(T lhs, const std::complex<T>& rhs)
{
    return lhs < rhs.real() ? true :
           lhs > rhs.real() ? false :
           T() < rhs.imag();
}

template <typename T>
struct real_type { typedef T type; };

template <typename T>
struct real_type<std::complex<T>> { typedef T type; };

template <typename T>
using real_type_t = typename real_type<T>::type;

template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
std::enable_if_t<!is_complex<T>::value,T>
conj(const T& val)
{
    return val;
}

template <typename T>
std::enable_if_t<is_complex<T>::value,T>
conj(const T& val)
{
    return std::conj(val);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value,T>
conj(bool, const T& val)
{
    return val;
}

template <typename T>
std::enable_if_t<is_complex<T>::value,T>
conj(bool cond, const T& val)
{
    return cond ? std::conj(val) : val;
}

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
}

}

#endif //TBLIS_INTERNAL_TYPES_HPP
