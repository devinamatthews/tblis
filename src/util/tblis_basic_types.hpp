#ifndef _TBLIS_BASIC_TYPES_HPP_
#define _TBLIS_BASIC_TYPES_HPP_

#include "tblis_config.hpp"

#include <cstdint>
#include <cstddef>

#include "external/stl_ext/include/complex.hpp"

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

enum type_t
{
    TYPE_SINGLE   = 0,
    TYPE_FLOAT    = TYPE_SINGLE,
    TYPE_DOUBLE   = 1,
    TYPE_SCOMPLEX = 2,
    TYPE_DCOMPLEX = 3
};

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

template <typename T> struct type_tag;
template <> struct type_tag<   float> { static constexpr type_t value =    TYPE_FLOAT; };
template <> struct type_tag<  double> { static constexpr type_t value =   TYPE_DOUBLE; };
template <> struct type_tag<scomplex> { static constexpr type_t value = TYPE_SCOMPLEX; };
template <> struct type_tag<dcomplex> { static constexpr type_t value = TYPE_DCOMPLEX; };

struct single_t;
constexpr single_t single;

//TODO configure these
typedef ssize_t len_type;
typedef ptrdiff_t stride_type;

using stl_ext::enable_if_t;
using stl_ext::enable_if_integral_t;
using stl_ext::enable_if_floating_point_t;

using stl_ext::real;
using stl_ext::imag;
using stl_ext::conj;
using stl_ext::real_type_t;
using stl_ext::complex_type_t;
using stl_ext::enable_if_complex_t;
using stl_ext::norm2;

}

#endif
