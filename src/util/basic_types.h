#ifndef _TBLIS_BASIC_TYPES_H_
#define _TBLIS_BASIC_TYPES_H_

#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#include "../tblis_config.h"

#include "assert.h"

#ifdef __cplusplus

#include <string>

#include "../memory/aligned_allocator.hpp"

#include "../external/stl_ext/include/complex.hpp"

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#undef assert
#define assert TBLIS_ASSERT
#include "../external/marray/include/varray.hpp"
#include "../external/marray/include/marray.hpp"

#else

#include <complex.h>

#endif

#ifdef __cplusplus
namespace tblis {
#endif

typedef struct tblis_config_s tblis_config;

typedef enum
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
} reduce_t;

typedef enum
{
    TYPE_SINGLE   = 0,
    TYPE_FLOAT    = TYPE_SINGLE,
    TYPE_DOUBLE   = 1,
    TYPE_SCOMPLEX = 2,
    TYPE_DCOMPLEX = 3
} type_t;

typedef TBLIS_LEN_TYPE len_type;
typedef TBLIS_STRIDE_TYPE stride_type;
typedef TBLIS_LABEL_TYPE label_type;

#ifdef __cplusplus

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

template <typename T> struct type_tag;
template <> struct type_tag<   float> { static constexpr type_t value =    TYPE_FLOAT; };
template <> struct type_tag<  double> { static constexpr type_t value =   TYPE_DOUBLE; };
template <> struct type_tag<scomplex> { static constexpr type_t value = TYPE_SCOMPLEX; };
template <> struct type_tag<dcomplex> { static constexpr type_t value = TYPE_DCOMPLEX; };

struct single_t
{
    constexpr single_t() {}
};
constexpr single_t single;

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
using stl_ext::is_complex;

template <typename T>
T conj(bool conjugate, T val)
{
    return (conjugate ? conj(val) : val);
}

template <typename T>
using const_tensor_view = MArray::const_varray_view<T>;

template <typename T>
using tensor_view = MArray::varray_view<T>;

template <typename T, typename Allocator=aligned_allocator<T,64>>
using tensor = MArray::varray<T, Allocator>;

using MArray::const_marray_view;
using MArray::marray_view;

template <typename T, unsigned ndim, typename Allocator=aligned_allocator<T,64>>
using marray = MArray::marray<T, ndim, Allocator>;

using MArray::const_matrix_view;
using MArray::matrix_view;

template <typename T, typename Allocator=aligned_allocator<T,64>>
using matrix = MArray::matrix<T, Allocator>;

using MArray::const_row_view;
using MArray::row_view;

template <typename T, typename Allocator=aligned_allocator<T,64>>
using row = MArray::row<T, Allocator>;

using MArray::Layout;
using MArray::COLUMN_MAJOR;
using MArray::ROW_MAJOR;
using MArray::DEFAULT;

using MArray::uninitialized_t;
using MArray::uninitialized;

using MArray::make_array;
using MArray::make_vector;

using MArray::range_t;
using MArray::range;

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
}

#else

typedef complex float scomplex;
typedef complex double dcomplex;

#endif

typedef struct tblis_scalar
{
    type_t type;
    char data[16] __attribute__((__aligned__(8)));

#ifdef __cplusplus

    tblis_scalar() : type(TYPE_DOUBLE) {}

    template <typename T>
    tblis_scalar(T value)
    : type(type_tag<T>::value)
    {
        *reinterpret_cast<T*>(data) = value;
    }

    template <typename T>
    T& get()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(data);
    }

    template <typename T>
    const T& get() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(data);
    }

#endif

} tblis_scalar;

typedef struct tblis_vector
{
    type_t type;
    int conj;
    char scalar[16] __attribute__((__aligned__(8)));
    void* data;
    len_type n;
    stride_type inc;

#ifdef __cplusplus

    tblis_vector()
    : type(TYPE_DOUBLE), conj(false), data(0), n(0), inc(0) {}

    template <typename T>
    tblis_vector(const T* A, len_type n, stride_type inc)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      n(n), inc(inc)
    {
        *reinterpret_cast<T*>(scalar) = T(1);
    }

    template <typename T>
    tblis_vector(T alpha, const T* A, len_type n, stride_type inc)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      n(n), inc(inc)
    {
        *reinterpret_cast<T*>(scalar) = alpha;
    }

    template <typename T>
    tblis_vector(const_row_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride())
    {
        *reinterpret_cast<T*>(scalar) = T(1);
    }

    template <typename T>
    tblis_vector(T alpha, const_row_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride())
    {
        *reinterpret_cast<T*>(scalar) = alpha;
    }

    template <typename T>
    T& alpha()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(scalar);
    }

    template <typename T>
    const T& alpha() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(scalar);
    }

    void swap(tblis_vector& other)
    {
        using std::swap;
        swap(type, other.type);
        swap(conj, other.conj);
        swap(scalar, other.scalar);
        swap(data, other.data);
        swap(n, other.n);
        swap(inc, other.inc);
    }

#endif

} tblis_vector;

typedef struct tblis_matrix
{
    type_t type;
    int conj;
    char scalar[16] __attribute__((__aligned__(8)));
    void* data;
    len_type m, n;
    stride_type rs, cs;

#ifdef __cplusplus

    tblis_matrix()
    : type(TYPE_DOUBLE), conj(false), data(0), m(0), n(0), rs(0), cs(0) {}

    template <typename T>
    tblis_matrix(const T* A, len_type m, len_type n, stride_type rs, stride_type cs)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      m(m), n(n), rs(rs), cs(rs)
    {
        *reinterpret_cast<T*>(scalar) = T(1);
    }

    template <typename T>
    tblis_matrix(T alpha, const T* A, len_type m, len_type n, stride_type rs, stride_type cs)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      m(m), n(n), rs(rs), cs(rs)
    {
        *reinterpret_cast<T*>(scalar) = alpha;
    }

    template <typename T>
    tblis_matrix(const_matrix_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1))
    {
        *reinterpret_cast<T*>(scalar) = T(1);
    }

    template <typename T>
    tblis_matrix(T alpha, const_matrix_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1))
    {
        *reinterpret_cast<T*>(scalar) = alpha;
    }

    template <typename T>
    T& alpha()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(scalar);
    }

    template <typename T>
    const T& alpha() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(scalar);
    }

    void swap(tblis_matrix& other)
    {
        using std::swap;
        swap(type, other.type);
        swap(conj, other.conj);
        swap(scalar, other.scalar);
        swap(data, other.data);
        swap(m, other.m);
        swap(n, other.n);
        swap(rs, other.rs);
        swap(cs, other.cs);
    }

#endif

} tblis_matrix;

typedef struct tblis_tensor
{
    type_t type;
    int conj;
    char scalar[16] __attribute__((__aligned__(8)));
    void* data;
    unsigned ndim;
    len_type* len;
    stride_type* stride;

#ifdef __cplusplus

    tblis_tensor() {}

    template <typename T>
    tblis_tensor(const_tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data()))
    {
        *reinterpret_cast<T*>(scalar) = T(1);
    }

    template <typename T>
    tblis_tensor(tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(view.data())),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data()))
    {
        *reinterpret_cast<T*>(scalar) = T(1);
    }

    template <typename T>
    tblis_tensor(T alpha, const_tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data()))
    {
        *reinterpret_cast<T*>(scalar) = alpha;
    }

    template <typename T>
    tblis_tensor(T alpha, tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(view.data())),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data()))
    {
        *reinterpret_cast<T*>(scalar) = alpha;
    }

    template <typename T>
    T& alpha()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(scalar);
    }

    template <typename T>
    const T& alpha() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(scalar);
    }

    void swap(tblis_tensor& other)
    {
        using std::swap;
        swap(type, other.type);
        swap(conj, other.conj);
        swap(scalar, other.scalar);
        swap(data, other.data);
        swap(ndim, other.ndim);
        swap(len, other.len);
        swap(stride, other.stride);
    }

#endif

} tblis_tensor;

#ifdef __cplusplus
}
#endif

#endif
