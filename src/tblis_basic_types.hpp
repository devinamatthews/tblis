#ifndef _TBLIS_BASIC_TYPES_HPP_
#define _TBLIS_BASIC_TYPES_HPP_

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

using idx_type = MArray::varray<float>::idx_type;
using stride_type = MArray::varray<float>::stride_type;

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

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
