#ifndef _TBLIS_BASIC_TYPES_H_
#define _TBLIS_BASIC_TYPES_H_

#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include "../tblis_config.h"

#ifdef __cplusplus
#include <complex>
#elif __STDC_VERSION__ >= 199901l
#include <complex.h>
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

#include <string>

#include "../memory/aligned_allocator.hpp"

#include "../external/stl_ext/include/complex.hpp"

#ifdef TBLIS_DEBUG
#define MARRAY_ENABLE_ASSERTS
#endif

#define MARRAY_LEN_TYPE TBLIS_LEN_TYPE
#define MARRAY_STRIDE_TYPE TBLIS_STRIDE_TYPE

#include "../external/marray/include/varray.hpp"
#include "../external/marray/include/marray.hpp"
#include "../external/marray/include/dpd_varray.hpp"
#include "../external/marray/include/dpd_marray.hpp"
#include "../external/marray/include/indexed_varray.hpp"
#include "../external/marray/include/indexed_dpd_varray.hpp"

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

namespace detail
{

template <typename T> struct label_vector_type
{ typedef MArray::short_vector<T,8> type; };

template <> struct label_vector_type<char>
{ typedef std::string type; };

}

typedef typename detail::label_vector_type<label_type>::type label_vector;

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

using namespace MArray;
namespace detail { using namespace MArray::detail; }
namespace slice { using namespace MArray::slice; }

template <typename T, typename Allocator=std::allocator<T>>
using tensor = MArray::varray<T, Allocator>;

template <typename T> struct type_tag;
template <> struct type_tag<   float> { static constexpr type_t value =    TYPE_FLOAT; };
template <> struct type_tag<  double> { static constexpr type_t value =   TYPE_DOUBLE; };
template <> struct type_tag<scomplex> { static constexpr type_t value = TYPE_SCOMPLEX; };
template <> struct type_tag<dcomplex> { static constexpr type_t value = TYPE_DCOMPLEX; };

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

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
}

#endif

typedef struct tblis_scalar
{
    union scalar
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

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)
        scalar(float    v) : s(v) {}
        scalar(double   v) : d(v) {}
        scalar(scomplex v) : c(v) {}
        scalar(dcomplex v) : z(v) {}
#endif
    } data;
    type_t type;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_scalar() : data(1.0), type(TYPE_DOUBLE) {}

    template <typename T>
    tblis_scalar(T value)
    : data(value), type(type_tag<T>::value) {}

    template <typename T>
    T& get();

    template <typename T>
    const T& get() const
    {
        return const_cast<tblis_scalar&>(*this).get<T>();
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

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <> inline
float& tblis_scalar::get<float>() { return data.s; }

template <> inline
double& tblis_scalar::get<double>() { return data.d; }

template <> inline
scomplex& tblis_scalar::get<scomplex>() { return data.c; }

template <> inline
dcomplex& tblis_scalar::get<dcomplex>() { return data.z; }

#endif

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
    tblis_scalar scalar;
    void* data;
    len_type n;
    stride_type inc;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_vector()
    : type(TYPE_DOUBLE), conj(false), data(0), n(0), inc(0), scalar(1.0) {}

    template <typename T>
    tblis_vector(const T* A, len_type n, stride_type inc)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      n(n), inc(inc), scalar(T(1)) {}

    template <typename T>
    tblis_vector(T alpha, const T* A, len_type n, stride_type inc)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      n(n), inc(inc), scalar(alpha) {}

    template <typename T>
    tblis_vector(const row_view<const T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride()), scalar(T(1)) {}

    template <typename T>
    tblis_vector(const row_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride()), scalar(T(1)) {}

    template <typename T>
    tblis_vector(T alpha, const row_view<const T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride()), scalar(alpha) {}

    template <typename T>
    tblis_vector(T alpha, const row_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      n(view.length()), inc(view.stride()), scalar(alpha) {}

    template <typename T>
    T& alpha()
    {
        return scalar.get<T>();
    }

    template <typename T>
    const T& alpha() const
    {
        return scalar.get<T>();
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
    tblis_scalar scalar;
    void* data;
    len_type m, n;
    stride_type rs, cs;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_matrix()
    : type(TYPE_DOUBLE), conj(false), data(0), m(0), n(0), rs(0), cs(0), scalar(1.0) {}

    template <typename T>
    tblis_matrix(const T* A, len_type m, len_type n, stride_type rs, stride_type cs)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      m(m), n(n), rs(rs), cs(rs), scalar(T(1)) {}

    template <typename T>
    tblis_matrix(T alpha, const T* A, len_type m, len_type n, stride_type rs, stride_type cs)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(A))),
      m(m), n(n), rs(rs), cs(rs), scalar(alpha) {}

    template <typename T>
    tblis_matrix(const matrix_view<const T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1)), scalar(T(1)) {}

    template <typename T>
    tblis_matrix(const matrix_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1)), scalar(T(1)) {}

    template <typename T>
    tblis_matrix(T alpha, const matrix_view<const T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1)), scalar(alpha) {}

    template <typename T>
    tblis_matrix(T alpha, const matrix_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      m(view.length(0)), n(view.length(1)), rs(view.stride(0)), cs(view.stride(1)), scalar(alpha) {}

    template <typename T>
    T& alpha()
    {
        return scalar.get<T>();
    }

    template <typename T>
    const T& alpha() const
    {
        return scalar.get<T>();
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
    tblis_scalar scalar;
    void* data;
    unsigned ndim;
    len_type* len;
    stride_type* stride;

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

    tblis_tensor()
    : type(TYPE_DOUBLE), conj(false), scalar(1.0), data(nullptr), ndim(0), len(nullptr),
      stride(nullptr) {}

    template <typename T>
    tblis_tensor(varray_view<const T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar(T(1)) {}

    template <typename T>
    tblis_tensor(varray_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar(T(1)) {}

    template <typename T>
    tblis_tensor(T alpha, varray_view<const T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar(alpha) {}

    template <typename T>
    tblis_tensor(T alpha, varray_view<T>& view)
    : type(type_tag<T>::value), conj(false), data(static_cast<void*>(const_cast<T*>(view.data()))),
      ndim(view.dimension()), len(const_cast<len_type*>(view.lengths().data())),
      stride(const_cast<stride_type*>(view.strides().data())), scalar(alpha) {}

    template <typename T>
    T& alpha()
    {
        return scalar.get<T>();
    }

    template <typename T>
    const T& alpha() const
    {
        return scalar.get<T>();
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

#define TBLIS_STRINGIZE_(...) #__VA_ARGS__
#define TBLIS_STRINGIZE(...) TBLIS_STRINGIZE_(__VA_ARGS__)
#define TBLIS_CONCAT_(x,y) x##y
#define TBLIS_CONCAT(x,y) TBLIS_CONCAT_(x,y)
#define TBLIS_FIRST_ARG(arg,...) arg

inline void __attribute__((format(printf, 2, 3),noreturn))
tblis_abort_with_message(const char* cond, const char* fmt, ...)
{
    if (strlen(fmt) == 1)
    {
        fprintf(stderr, "%s\n", cond);
    }
    else
    {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
    }
    abort();
}

#ifdef TBLIS_DEBUG

#define TBLIS_ASSERT(x,...) ((x) ? (void)(x) : \
    tblis_abort_with_message(TBLIS_STRINGIZE(x), "" __VA_ARGS__ "\n"))

#else

#define TBLIS_ASSERT(...) ((void)0)

#endif

#endif
