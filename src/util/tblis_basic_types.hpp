#ifndef _TBLIS_BASIC_TYPES_HPP_
#define _TBLIS_BASIC_TYPES_HPP_

#include <cstddef>

#include "external/stl_ext/include/complex.hpp"

namespace tblis
{

typedef ssize_t idx_type;
typedef ptrdiff_t stride_type;

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
