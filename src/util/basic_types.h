#ifndef _TBLIS_BASIC_TYPES_H_
#define _TBLIS_BASIC_TYPES_H_

#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#include "../tblis_config.h"

#include "assert.h"

#ifdef __cplusplus
#include <complex>
#elif __STDC_VERSION__ >= 199901l
#include <complex.h>
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

#include <string>

#include "../memory/aligned_allocator.hpp"

#include "../external/stl_ext/include/complex.hpp"

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#undef assert
#define assert TBLIS_ASSERT
#include "../external/marray/include/varray.hpp"
#include "../external/marray/include/marray.hpp"

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

typedef struct fake_scomplex { float real, imag; } fake_scomplex;
typedef struct fake_dcomplex { double real, imag; } fake_dcomplex;

#ifdef __cplusplus

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

#elif __STDC_VERSION__ >= 199901l

typedef complex float scomplex;
typedef complex double dcomplex;

#else

typedef fake_scomplex scomplex;
typedef fake_dcomplex dcomplex;

#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

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

#endif

typedef union scalar_variant
{
    float s;
    double d;
#if defined(__cplusplus) && defined(TBLIS_DONT_USE_CXX11)
    fake_scomplex c;
    fake_dcomplex z;
#else
    scomplex c;
    dcomplex z;
#endif
} scalar_variant;

typedef struct tblis_scalar
{
    type_t type;
    scalar_variant data;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_scalar() : type(TYPE_DOUBLE), data{} {}

    template <typename T>
    tblis_scalar(T value)
    : type(type_tag<T>::value), data{}
    {
        *reinterpret_cast<T*>(&data) = value;
    }

    template <typename T>
    T& get()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(&data);
    }

    template <typename T>
    const T& get() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(&data);
    }

    void swap(tblis_scalar& other)
    {
        using std::swap;
        swap(type, other.type);
        swap(data, other.data);
    }

    friend void swap(tblis_scalar& a, tblis_scalar& b)
    {
        a.swap(b);
    }

#endif

} tblis_scalar;

#ifdef __cplusplus
extern "C"
{
#endif

void tblis_init_scalar_s(tblis_scalar* s, float value);

void tblis_init_scalar_d(tblis_scalar* s, double value);

void tblis_init_scalar_c(tblis_scalar* s, scomplex value);

void tblis_init_scalar_z(tblis_scalar* s, dcomplex value);

#ifdef __cplusplus
}
#endif

typedef struct tblis_vector
{
    type_t type;
    int conj;
    scalar_variant scalar;
    void* data;
    len_type n;
    stride_type inc;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_vector()
    : type(TYPE_DOUBLE), conj(false), data(0), n(0), inc(0), scalar{} {}

    template <typename T>
    tblis_vector(const T* A, len_type n, stride_type inc)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      n(n), inc(inc), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = T(1);
    }

    template <typename T>
    tblis_vector(T alpha, const T* A, len_type n, stride_type inc)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      n(n), inc(inc), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = alpha;
    }

    template <typename T>
    tblis_vector(const_row_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride()), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = T(1);
    }

    template <typename T>
    tblis_vector(T alpha, const_row_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride()), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = alpha;
    }

    template <typename T>
    T& alpha()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(&scalar);
    }

    template <typename T>
    const T& alpha() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(&scalar);
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

    friend void swap(tblis_vector& a, tblis_vector& b)
    {
        a.swap(b);
    }

#endif

} tblis_vector;

#ifdef __cplusplus
extern "C"
{
#endif

void tblis_init_vector_scaled_s(tblis_vector* v, float scalar,
                                len_type n, float* data,stride_type inc);

void tblis_init_vector_scaled_d(tblis_vector* v, double scalar,
                                len_type n, double* data,stride_type inc);

void tblis_init_vector_scaled_c(tblis_vector* v, scomplex scalar,
                                len_type n, scomplex* data,stride_type inc);

void tblis_init_vector_scaled_z(tblis_vector* v, dcomplex scalar,
                                len_type n, dcomplex* data,stride_type inc);

void tblis_init_vector_s(tblis_vector* v,
                         len_type n, float* data,stride_type inc);

void tblis_init_vector_d(tblis_vector* v,
                         len_type n, double* data,stride_type inc);

void tblis_init_vector_c(tblis_vector* v,
                         len_type n, scomplex* data,stride_type inc);

void tblis_init_vector_z(tblis_vector* v,
                         len_type n, dcomplex* data,stride_type inc);

#ifdef __cplusplus
}
#endif

typedef struct tblis_matrix
{
    type_t type;
    int conj;
    scalar_variant scalar;
    void* data;
    len_type m, n;
    stride_type rs, cs;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_matrix()
    : type(TYPE_DOUBLE), conj(false), data(0), m(0), n(0), rs(0), cs(0), scalar{} {}

    template <typename T>
    tblis_matrix(const T* A, len_type m, len_type n, stride_type rs, stride_type cs)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      m(m), n(n), rs(rs), cs(rs), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = T(1);
    }

    template <typename T>
    tblis_matrix(T alpha, const T* A, len_type m, len_type n, stride_type rs, stride_type cs)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      m(m), n(n), rs(rs), cs(rs), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = alpha;
    }

    template <typename T>
    tblis_matrix(const_matrix_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1)), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = T(1);
    }

    template <typename T>
    tblis_matrix(T alpha, const_matrix_view<T> view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1)), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = alpha;
    }

    template <typename T>
    T& alpha()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(&scalar);
    }

    template <typename T>
    const T& alpha() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(&scalar);
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

    friend void swap(tblis_matrix& a, tblis_matrix& b)
    {
        a.swap(b);
    }

#endif

} tblis_matrix;

#ifdef __cplusplus
extern "C"
{
#endif

void tblis_init_matrix_scaled_s(tblis_matrix* mat, float scalar,
                                len_type m, len_type n, float* data,
                                stride_type rs, stride_type cs);

void tblis_init_matrix_scaled_d(tblis_matrix* mat, double scalar,
                                len_type m, len_type n, double* data,
                                stride_type rs, stride_type cs);

void tblis_init_matrix_scaled_c(tblis_matrix* mat, scomplex scalar,
                                len_type m, len_type n, scomplex* data,
                                stride_type rs, stride_type cs);

void tblis_init_matrix_scaled_z(tblis_matrix* mat, dcomplex scalar,
                                len_type m, len_type n, dcomplex* data,
                                stride_type rs, stride_type cs);

void tblis_init_matrix_s(tblis_matrix* mat,
                         len_type m, len_type n, float* data,
                         stride_type rs, stride_type cs);

void tblis_init_matrix_d(tblis_matrix* mat,
                         len_type m, len_type n, double* data,
                         stride_type rs, stride_type cs);

void tblis_init_matrix_c(tblis_matrix* mat,
                         len_type m, len_type n, scomplex* data,
                         stride_type rs, stride_type cs);

void tblis_init_matrix_z(tblis_matrix* mat,
                         len_type m, len_type n, dcomplex* data,
                         stride_type rs, stride_type cs);

#ifdef __cplusplus
}
#endif

typedef struct tblis_tensor
{
    type_t type;
    int conj;
    scalar_variant scalar;
    void* data;
    unsigned ndim;
    len_type* len;
    stride_type* stride;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_tensor() : scalar{} {}

    template <typename T>
    tblis_tensor(const_tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = T(1);
    }

    template <typename T>
    tblis_tensor(tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(view.data())),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = T(1);
    }

    template <typename T>
    tblis_tensor(T alpha, const_tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = alpha;
    }

    template <typename T>
    tblis_tensor(T alpha, tensor_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(view.data())),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar{}
    {
        *reinterpret_cast<T*>(&scalar) = alpha;
    }

    template <typename T>
    T& alpha()
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<T*>(&scalar);
    }

    template <typename T>
    const T& alpha() const
    {
        TBLIS_ASSERT(type_tag<T>::value == type);
        return *reinterpret_cast<const T*>(&scalar);
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

    friend void swap(tblis_tensor& a, tblis_tensor& b)
    {
        a.swap(b);
    }

#endif

} tblis_tensor;

#ifdef __cplusplus
extern "C"
{
#endif

void tblis_init_tensor_scaled_s(tblis_tensor* t, float scalar,
                                unsigned ndim, len_type* len, float* data,
                                stride_type* stride);

void tblis_init_tensor_scaled_d(tblis_tensor* t, double scalar,
                                unsigned ndim, len_type* len, double* data,
                                stride_type* stride);

void tblis_init_tensor_scaled_c(tblis_tensor* t, scomplex scalar,
                                unsigned ndim, len_type* len, scomplex* data,
                                stride_type* stride);

void tblis_init_tensor_scaled_z(tblis_tensor* t, dcomplex scalar,
                                unsigned ndim, len_type* len, dcomplex* data,
                                stride_type* stride);

void tblis_init_tensor_s(tblis_tensor* t,
                         unsigned ndim, len_type* len, float* data,
                         stride_type* stride);

void tblis_init_tensor_d(tblis_tensor* t,
                         unsigned ndim, len_type* len, double* data,
                         stride_type* stride);

void tblis_init_tensor_c(tblis_tensor* t,
                         unsigned ndim, len_type* len, scomplex* data,
                         stride_type* stride);

void tblis_init_tensor_z(tblis_tensor* t,
                         unsigned ndim, len_type* len, dcomplex* data,
                         stride_type* stride);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
}
#endif

#endif
