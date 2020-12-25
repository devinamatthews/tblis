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

#define TBLIS_STRINGIZE_(...) #__VA_ARGS__
#define TBLIS_STRINGIZE(...) TBLIS_STRINGIZE_(__VA_ARGS__)
#define TBLIS_CONCAT_(x,y) x##y
#define TBLIS_CONCAT(x,y) TBLIS_CONCAT_(x,y)
#define TBLIS_FIRST_ARG(arg,...) arg

#ifdef __cplusplus

#include <type_traits>

inline void __attribute__((format(printf, 1, 2),noreturn))
tblis_abort_with_message(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

inline void tblis_check_assert(const char* cond_str, bool cond)
{
    if (__builtin_expect(!cond,0))
        tblis_abort_with_message("%s", cond_str);
}

template <typename... Args>
inline void tblis_check_assert(const char*, bool cond, const char* fmt, Args&&... args)
{
    if (__builtin_expect(!cond,0))
        tblis_abort_with_message(fmt, std::forward<Args>(args)...);
}

#ifdef TBLIS_DEBUG

#define TBLIS_ASSERT(...) \
    tblis_check_assert(TBLIS_STRINGIZE(TBLIS_FIRST_ARG(__VA_ARGS__,0)), __VA_ARGS__)

#else

#define TBLIS_ASSERT(...) ((void)0)

#endif

#endif

#ifdef __cplusplus
#define TBLIS_EXPORT extern "C"
#else
#define TBLIS_EXPORT
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

#include <string>
#include <type_traits>
#include <memory>

#include "../memory/aligned_allocator.hpp"
#include "../external/stl_ext/include/complex.hpp"

#if defined(TBLIS_DEBUG) && !defined(MARRAY_ENABLE_ASSERTS)
#define MARRAY_ENABLE_ASSERTS
#endif

#ifndef MARRAY_LEN_TYPE
#define MARRAY_LEN_TYPE TBLIS_LEN_TYPE
#endif

#ifndef MARRAY_STRIDE_TYPE
#define MARRAY_STRIDE_TYPE TBLIS_STRIDE_TYPE
#endif

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
#define TBLIS_MAX_UNROLL 8

#ifdef __cplusplus

namespace detail
{

template <typename T> struct label_vector_helper
{ typedef MArray::short_vector<T,MARRAY_OPT_NDIM> type; };

template <> struct label_vector_helper<char>
{ typedef std::string type; };

}

typedef typename detail::label_vector_helper<label_type>::type label_vector;

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

#else

typedef complex float scomplex;
typedef complex double dcomplex;

#endif

#if defined(__cplusplus)

#if !defined(TBLIS_DONT_USE_CXX11)

using namespace MArray;
namespace detail { using namespace MArray::detail; }
namespace slice { using namespace MArray::slice; }

#endif

template <typename T> struct type_tag { static constexpr type_t value =   TYPE_DOUBLE; };
template <> struct type_tag<   float> { static constexpr type_t value =    TYPE_FLOAT; };
template <> struct type_tag<  double> { static constexpr type_t value =   TYPE_DOUBLE; };
template <> struct type_tag<scomplex> { static constexpr type_t value = TYPE_SCOMPLEX; };
template <> struct type_tag<dcomplex> { static constexpr type_t value = TYPE_DCOMPLEX; };

constexpr std::array<size_t,4> type_size =
{
    sizeof(   float),
    sizeof(  double),
    sizeof(scomplex),
    sizeof(dcomplex),
};

constexpr std::array<size_t,4> type_alignment =
{
    alignof(   float),
    alignof(  double),
    alignof(scomplex),
    alignof(dcomplex),
};

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
        scomplex c;
        dcomplex z;

#if defined(__cplusplus)
        scalar() : z(0.0, 0.0) {}
#endif
    } data;
    type_t type;

#if defined(__cplusplus)

    /*
    tblis_scalar()
    : type(TYPE_DOUBLE)
    {
        data.d = 1.0;
    }
    */

    tblis_scalar(const tblis_scalar& other)
    : type(other.type)
    {
        memcpy(&data, &other.data, sizeof(data));
    }

    template <typename T>
    tblis_scalar(T value, type_t type = type_tag<T>::value)
    : type(type)
    {
        *this = value;
    }

    template <typename T>
    T& get();

    template <typename T>
    const T& get() const
    {
        return const_cast<tblis_scalar&>(*this).get<T>();
    }

    void reset(const tblis_scalar& other)
    {
        type = other.type;
        memcpy(&data, &other.data, sizeof(data));
    }

    template <typename T>
    void reset(T value, type_t type = type_tag<T>::value)
    {
        this->type = type;
        *this = value;
    }

    tblis_scalar& operator=(const tblis_scalar& other)
    {
        switch (type)
        {
            case TYPE_FLOAT:
                switch (other.type)
                {
                    case TYPE_FLOAT:    data.s = float(     other.data.s ); break;
                    case TYPE_DOUBLE:   data.s = float(     other.data.d ); break;
                    case TYPE_SCOMPLEX: data.s = float(real(other.data.c)); break;
                    case TYPE_DCOMPLEX: data.s = float(real(other.data.z)); break;
                }
                break;
            case TYPE_DOUBLE:
                switch (other.type)
                {
                    case TYPE_FLOAT:    data.d = double(     other.data.s ); break;
                    case TYPE_DOUBLE:   data.d = double(     other.data.d ); break;
                    case TYPE_SCOMPLEX: data.d = double(real(other.data.c)); break;
                    case TYPE_DCOMPLEX: data.d = double(real(other.data.z)); break;
                }
                break;
            case TYPE_SCOMPLEX:
                switch (other.type)
                {
                    case TYPE_FLOAT:    data.c = scomplex(other.data.s); break;
                    case TYPE_DOUBLE:   data.c = scomplex(other.data.d); break;
                    case TYPE_SCOMPLEX: data.c = scomplex(other.data.c); break;
                    case TYPE_DCOMPLEX: data.c = scomplex(other.data.z); break;
                }
                break;
            case TYPE_DCOMPLEX:
                switch (other.type)
                {
                    case TYPE_FLOAT:    data.z = dcomplex(other.data.s); break;
                    case TYPE_DOUBLE:   data.z = dcomplex(other.data.d); break;
                    case TYPE_SCOMPLEX: data.z = dcomplex(other.data.c); break;
                    case TYPE_DCOMPLEX: data.z = dcomplex(other.data.z); break;
                }
                break;
        }

        return *this;
    }

    tblis_scalar convert(type_t new_type) const
    {
        tblis_scalar ret(0, new_type);
        ret = *this;
        return ret;
    }

    const char* raw() const { return reinterpret_cast<const char*>(&data); }

    char* raw() { return reinterpret_cast<char*>(&data); }

    void to(char* buf) const
    {
        memcpy(buf, raw(), type_size[type]);
    }

    void from(const char* buf)
    {
        memcpy(raw(), buf, type_size[type]);
    }

    template <typename T>
    tblis_scalar& operator=(T value)
    {
        switch (type)
        {
            case TYPE_FLOAT:    data.s = float   (real(value)); break;
            case TYPE_DOUBLE:   data.d = double  (real(value)); break;
            case TYPE_SCOMPLEX: data.c = scomplex(     value ); break;
            case TYPE_DCOMPLEX: data.z = dcomplex(     value ); break;
        }

        return *this;
    }

    bool is_zero() const
    {
        switch (type)
        {
            case TYPE_FLOAT:    return data.s == 0.0f;
            case TYPE_DOUBLE:   return data.d == 0.0;
            case TYPE_SCOMPLEX: return data.c == 0.0f;
            case TYPE_DCOMPLEX: return data.z == 0.0;
        }

        return false;
    }

    bool is_negative() const
    {
        switch (type)
        {
            case TYPE_FLOAT:    return data.s < 0.0f;
            case TYPE_DOUBLE:   return data.d < 0.0;
            case TYPE_SCOMPLEX: return data.c.real() < 0.0f ||
                (data.c.real() == 0.0f && data.c.imag() < 0.0f);
            case TYPE_DCOMPLEX: return data.z.real() < 0.0 ||
                (data.z.real() == 0.0 && data.z.imag() < 0.0);
        }

        return false;
    }

    bool is_one() const
    {
        switch (type)
        {
            case TYPE_FLOAT:    return data.s == 1.0f;
            case TYPE_DOUBLE:   return data.d == 1.0;
            case TYPE_SCOMPLEX: return data.c == 1.0f;
            case TYPE_DCOMPLEX: return data.z == 1.0;
        }

        return false;
    }

    bool is_complex() const
    {
        switch (type)
        {
            case TYPE_FLOAT:    return false;
            case TYPE_DOUBLE:   return false;
            case TYPE_SCOMPLEX: return true;
            case TYPE_DCOMPLEX: return true;
        }

        return false;
    }

    tblis_scalar& operator+=(const tblis_scalar& other)
    {
        TBLIS_ASSERT(type == other.type);

        switch (type)
        {
            case TYPE_FLOAT:    data.s += other.data.s; break;
            case TYPE_DOUBLE:   data.d += other.data.d; break;
            case TYPE_SCOMPLEX: data.c += other.data.c; break;
            case TYPE_DCOMPLEX: data.z += other.data.z; break;
        }

        return *this;
    }

    tblis_scalar& operator-=(const tblis_scalar& other)
    {
        TBLIS_ASSERT(type == other.type);

        switch (type)
        {
            case TYPE_FLOAT:    data.s -= other.data.s; break;
            case TYPE_DOUBLE:   data.d -= other.data.d; break;
            case TYPE_SCOMPLEX: data.c -= other.data.c; break;
            case TYPE_DCOMPLEX: data.z -= other.data.z; break;
        }

        return *this;
    }

    tblis_scalar& operator*=(const tblis_scalar& other)
    {
        TBLIS_ASSERT(type == other.type);

        switch (type)
        {
            case TYPE_FLOAT:    data.s *= other.data.s; break;
            case TYPE_DOUBLE:   data.d *= other.data.d; break;
            case TYPE_SCOMPLEX: data.c *= other.data.c; break;
            case TYPE_DCOMPLEX: data.z *= other.data.z; break;
        }

        return *this;
    }

    tblis_scalar& operator/=(const tblis_scalar& other)
    {
        TBLIS_ASSERT(type == other.type);

        switch (type)
        {
            case TYPE_FLOAT:    data.s /= other.data.s; break;
            case TYPE_DOUBLE:   data.d /= other.data.d; break;
            case TYPE_SCOMPLEX: data.c /= other.data.c; break;
            case TYPE_DCOMPLEX: data.z /= other.data.z; break;
        }

        return *this;
    }

    tblis_scalar operator+(const tblis_scalar& other) const
    {
        tblis_scalar ret(*this);
        ret += other;
        return ret;
    }

    tblis_scalar operator-(const tblis_scalar& other) const
    {
        tblis_scalar ret(*this);
        ret -= other;
        return ret;
    }

    tblis_scalar operator*(const tblis_scalar& other) const
    {
        tblis_scalar ret(*this);
        ret *= other;
        return ret;
    }

    tblis_scalar operator/(const tblis_scalar& other) const
    {
        tblis_scalar ret(*this);
        ret /= other;
        return ret;
    }

    tblis_scalar& conj()
    {
        switch (type)
        {
            case TYPE_FLOAT:    break;
            case TYPE_DOUBLE:   break;
            case TYPE_SCOMPLEX: data.c = std::conj(data.c); break;
            case TYPE_DCOMPLEX: data.z = std::conj(data.z); break;
        }

        return *this;
    }

    friend tblis_scalar conj(const tblis_scalar& other)
    {
        tblis_scalar ret(other);
        ret.conj();
        return other;
    }

    tblis_scalar& abs()
    {
        switch (type)
        {
            case TYPE_FLOAT:    data.s = std::abs(data.s); break;
            case TYPE_DOUBLE:   data.d = std::abs(data.d); break;
            case TYPE_SCOMPLEX: data.c = std::abs(data.c); break;
            case TYPE_DCOMPLEX: data.z = std::abs(data.z); break;
        }

        return *this;
    }

    friend tblis_scalar abs(const tblis_scalar& other)
    {
        tblis_scalar ret(other);
        ret.abs();
        return other;
    }

    tblis_scalar& sqrt()
    {
        switch (type)
        {
            case TYPE_FLOAT:    data.s = std::sqrt(data.s); break;
            case TYPE_DOUBLE:   data.d = std::sqrt(data.d); break;
            case TYPE_SCOMPLEX: data.c = std::sqrt(data.c); break;
            case TYPE_DCOMPLEX: data.z = std::sqrt(data.z); break;
        }

        return *this;
    }

    friend tblis_scalar sqrt(const tblis_scalar& other)
    {
        tblis_scalar ret(other);
        ret.sqrt();
        return other;
    }

#endif

} tblis_scalar;

#if defined(__cplusplus)

template <> inline
float& tblis_scalar::get<float>() { return data.s; }

template <> inline
double& tblis_scalar::get<double>() { return data.d; }

template <> inline
scomplex& tblis_scalar::get<scomplex>() { return data.c; }

template <> inline
dcomplex& tblis_scalar::get<dcomplex>() { return data.z; }

#endif

TBLIS_EXPORT void tblis_init_scalar_s(tblis_scalar* s, float value);

TBLIS_EXPORT void tblis_init_scalar_d(tblis_scalar* s, double value);

TBLIS_EXPORT void tblis_init_scalar_c(tblis_scalar* s, scomplex value);

TBLIS_EXPORT void tblis_init_scalar_z(tblis_scalar* s, dcomplex value);

typedef struct tblis_tensor
{
    type_t type;
    int conj;
    tblis_scalar scalar;
    void* data;
    int ndim;
    len_type* len;
    stride_type* stride;

#if defined(__cplusplus)

    tblis_tensor()
    : type(TYPE_DOUBLE), conj(false), scalar(1.0), data(0),
      ndim(0), len(0), stride(0) {}

    template <typename T>
    tblis_tensor(const T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : type(type_tag<T>::value), conj(false), scalar(T(1)),
      data(const_cast<T*>(A)), ndim(ndim), len(const_cast<len_type*>(len)),
      stride(const_cast<stride_type*>(stride)) {}

    template <typename T>
    tblis_tensor(T alpha, const T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : type(type_tag<T>::value), conj(false), scalar(alpha),
      data(const_cast<T*>(A)), ndim(ndim), len(const_cast<len_type*>(len)),
      stride(const_cast<stride_type*>(stride)) {}

    template <typename T>
    tblis_tensor(T alpha, bool conj, const T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : type(type_tag<T>::value), conj(conj), scalar(alpha),
      data(const_cast<T*>(A)), ndim(ndim), len(const_cast<len_type*>(len)),
      stride(const_cast<stride_type*>(stride)) {}

#endif

} tblis_tensor;

TBLIS_EXPORT void tblis_init_tensor_scaled_s(tblis_tensor* t, float scalar,
                                             int ndim, len_type* len, float* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_scaled_d(tblis_tensor* t, double scalar,
                                             int ndim, len_type* len, double* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_scaled_c(tblis_tensor* t, scomplex scalar,
                                             int ndim, len_type* len, scomplex* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_scaled_z(tblis_tensor* t, dcomplex scalar,
                                             int ndim, len_type* len, dcomplex* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_s(tblis_tensor* t,
                                      int ndim, len_type* len, float* data,
                                      stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_d(tblis_tensor* t,
                                      int ndim, len_type* len, double* data,
                                      stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_c(tblis_tensor* t,
                                      int ndim, len_type* len, scomplex* data,
                                      stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_z(tblis_tensor* t,
                                      int ndim, len_type* len, dcomplex* data,
                                      stride_type* stride);

#ifdef __cplusplus

using scalar = tblis_scalar;

#if defined(TBLIS_DONT_USE_CXX11)

using tensor = tblis_tensor;

#else

struct tensor : tblis_tensor
{
    len_vector len_buf;
    stride_vector stride_buf;

    template <typename T, int N, typename D, bool O>
    tensor(const marray_base<T,N,D,O>& t)
    : tblis_tensor(t.data(), t.dimension(),
                   t.lengths().data(), t.strides().data()) {}

    template <typename T, typename D, bool O>
    tensor(const varray_base<T,D,O>& t)
    : tblis_tensor(t.data(), t.dimension(),
                   t.lengths().data(), t.strides().data()) {}

    template <typename T, int N, int I, typename... D>
    tensor(const marray_slice<T,N,I,D...>& t)
    : tensor(t.view()) {}

#if defined(EIGEN_CXX11_TENSOR_TENSOR_H)

    template <typename T, int N, int O, typename I>
    tensor(const Eigen::Tensor<T,N,O,I>& t)
    : tblis_tensor(t.data(), N, nullptr, nullptr)
    {
        auto dims = t.dimensions();
        len_buf.assign(dims.begin(), dims.end());
        stride_buf = varray<T>::strides(len_buf, t.Options&Eigen::RowMajor ? ROW_MAJOR : COLUMN_MAJOR);
        len = len_buf.data();
        stride = stride_buf.data();
    }

#endif

#if defined(EIGEN_CXX11_TENSOR_TENSOR_FIXED_SIZE_H)

    template <typename T, typename D, int O, typename I>
    tensor(const Eigen::TensorFixedSize<T,D,O,I>& t)
    : tblis_tensor(t.data(), t.NumIndices, nullptr, nullptr)
    {
        auto dims = t.dimensions();
        len_buf.assign(dims.begin(), dims.end());
        stride_buf = varray<T>::strides(len_buf, t.Options&Eigen::RowMajor ? ROW_MAJOR : COLUMN_MAJOR);
        len = len_buf.data();
        stride = stride_buf.data();
    }

#endif

#if defined(EIGEN_CXX11_TENSOR_TENSOR_MAP_H)

    template <typename Tensor, int O, template <class> class MP>
    tensor(const Eigen::TensorMap<Tensor,O,MP>& t)
    : tblis_tensor(t.data(), t.NumIndices, nullptr, nullptr)
    {
        auto dims = t.dimensions();
        len_buf.assign(dims.begin(), dims.end());
        stride_buf = varray<double>::strides(len_buf, Tensor::Options&Eigen::RowMajor ? ROW_MAJOR : COLUMN_MAJOR);
        len = len_buf.data();
        stride = stride_buf.data();
    }

#endif

};

inline label_vector idx(const tblis_tensor& A, label_vector&& = label_vector())
{
    return range(A.ndim);
}

label_vector idx(const std::string& from, label_vector&& to = label_vector());

#endif

}
#endif

#endif
