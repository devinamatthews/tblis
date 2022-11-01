#ifndef TBLIS_BASE_TYPES_H
#define TBLIS_BASE_TYPES_H 1

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <tblis/config.h>
#include <tblis/base/macros.h>

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

#if TBLIS_ENABLE_CXX
#include <array>
#include <vector>
#endif

TBLIS_BEGIN_NAMESPACE

TBLIS_DEFINE_ENUM
    (REDUCE_SUM     , 0)
    (REDUCE_SUM_ABS , 1)
    (REDUCE_NORM_1  , 1)
    (REDUCE_MAX     , 2)
    (REDUCE_MAX_ABS , 3)
    (REDUCE_NORM_INF, 3)
    (REDUCE_MIN     , 4)
    (REDUCE_MIN_ABS , 5)
    (REDUCE_NORM_2  , 6)
(reduce_t)

TBLIS_DEFINE_ENUM
    (SINGLE  , 0)
    (FLOAT   , 0)
    (DOUBLE  , 1)
    (SCOMPLEX, 2)
    (COMPLEX , 2)
    (DCOMPLEX, 3)
(type_t)

typedef TBLIS_LEN_TYPE tblis_len_type;
typedef TBLIS_STRIDE_TYPE tblis_stride_type;
typedef TBLIS_LABEL_TYPE tblis_label_type;
#define TBLIS_MAX_UNROLL 8

#if TBLIS_ENABLE_CXX
using len_type = tblis_len_type;
using stride_type = tblis_stride_type;
using label_type = tblis_label_type;
#endif

#ifndef TBLIS_OPT_NDIM
#define TBLIS_OPT_NDIM 8
#endif

#ifdef __cplusplus

typedef std::complex<float> tblis_scomplex;
typedef std::complex<double> tblis_dcomplex;

#if TBLIS_ENABLE_CXX
using scomplex = tblis_scomplex;
using dcomplex = tblis_dcomplex;
#endif

#else

typedef complex float tblis_scomplex;
typedef complex double tblis_dcomplex;

#endif //__cplusplus

#if TBLIS_ENABLE_CXX

template <typename T> struct type_tag { static constexpr type_t value =   DOUBLE; };
template <> struct type_tag<   float> { static constexpr type_t value =    FLOAT; };
template <> struct type_tag<  double> { static constexpr type_t value =   DOUBLE; };
template <> struct type_tag<scomplex> { static constexpr type_t value = SCOMPLEX; };
template <> struct type_tag<dcomplex> { static constexpr type_t value = DCOMPLEX; };

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

#endif //TBLIS_ENABLE_CXX

typedef struct tblis_scalar
{
    union scalar
    {
        float s;
        double d;
        tblis_scomplex c;
        tblis_dcomplex z;
#ifdef __cplusplus
        scalar() : z{} {}
#endif
    } data;
    tblis_type_t type;

#if TBLIS_ENABLE_CXX

    template <typename T> struct is_scalar : std::is_arithmetic<T> {};

    template <typename T> struct is_scalar<std::complex<T>> : std::is_arithmetic<T> {};

    template <> struct is_scalar<scomplex> : std::true_type {};

    template <> struct is_scalar<dcomplex> : std::true_type {};

    tblis_scalar(const tblis_scalar& other)
    : type(other.type)
    {
        memcpy(&data, &other.data, sizeof(data));
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    tblis_scalar(T value, type_t type = type_tag<T>::value)
    : type(type)
    {
        *this = value;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    T get() const;

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    void set(T val)
    {
        if (type & DOUBLE) data.z = val;
        else               data.c = val;
    }

    void reset(const tblis_scalar& other)
    {
        type = other.type;
        memcpy(&data, &other.data, sizeof(data));
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    void reset(T value, type_t type = type_tag<T>::value)
    {
        this->type = type;
        *this = value;
    }

    tblis_scalar& operator=(const tblis_scalar& other)
    {
        if (TBLIS_LIKELY((type & DOUBLE) == (other.type & DOUBLE)))
        {
            memcpy(&data, &other.data, sizeof(data));
        }
        else if (type & DOUBLE)
        {
            data.z = other.data.c;
        }
        else
        {
            data.c = other.data.z;
        }

        return *this;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    tblis_scalar& operator=(T other)
    {
        set(other);
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

    bool is_zero() const
    {
        switch (type)
        {
            case FLOAT:    return data.s == 0.0f;
            case DOUBLE:   return data.d == 0.0;
            case SCOMPLEX: return data.c == 0.0f;
            case DCOMPLEX: return data.z == 0.0;
        }

        return false;
    }

    bool is_negative() const
    {
        switch (type)
        {
            case FLOAT:    return data.s < 0.0f;
            case DOUBLE:   return data.d < 0.0;
            case SCOMPLEX: return data.c.real() < 0.0f ||
                (data.c.real() == 0.0f && data.c.imag() < 0.0f);
            case DCOMPLEX: return data.z.real() < 0.0 ||
                (data.z.real() == 0.0 && data.z.imag() < 0.0);
        }

        return false;
    }

    bool is_one() const
    {
        switch (type)
        {
            case FLOAT:    return data.s == 1.0f;
            case DOUBLE:   return data.d == 1.0;
            case SCOMPLEX: return data.c == 1.0f;
            case DCOMPLEX: return data.z == 1.0;
        }

        return false;
    }

    bool is_complex() const
    {
        switch (type)
        {
            case FLOAT:    return false;
            case DOUBLE:   return false;
            case SCOMPLEX: return true;
            case DCOMPLEX: return true;
        }

        return false;
    }

    template <typename T>
    std::enable_if_t<type_tag<T>::value >= 0,tblis_scalar&>
    operator+=(T other)
    {
        if (type & DOUBLE) data.z += other;
        else               data.c += other;

        return *this;
    }

    template <typename T>
    std::enable_if_t<type_tag<T>::value >= 0,tblis_scalar&>
    operator-=(T other)
    {
        if (type & DOUBLE) data.z -= other;
        else               data.c -= other;

        return *this;
    }

    template <typename T>
    std::enable_if_t<!!(type_tag<T>::value & SCOMPLEX), tblis_scalar&>
    operator*=(T other)
    {
        switch (type)
        {
            case    FLOAT: data.s *= std::real(other); break;
            case   DOUBLE: data.d *= std::real(other); break;
            case SCOMPLEX: data.c = T(data.c) *          other ; break;
            case DCOMPLEX: data.z =   data.z  * dcomplex(other); break;
        }

        return *this;
    }

    template <typename T>
    std::enable_if_t<!(type_tag<T>::value & SCOMPLEX), tblis_scalar&>
    operator*=(T other)
    {
        switch (type)
        {
            case    FLOAT: data.s *= other; break;
            case   DOUBLE: data.d *= other; break;
            case SCOMPLEX: data.c = std::complex<T>(data.c) *        other ; break;
            case DCOMPLEX: data.z =                 data.z  * double(other); break;
        }

        return *this;
    }

    template <typename T>
    std::enable_if_t<!!(type_tag<T>::value & SCOMPLEX), tblis_scalar&>
    operator/=(T other)
    {
        switch (type)
        {
            case    FLOAT: data.s = std::real(T(data.s) /          other ); break;
            case   DOUBLE: data.d = std::real(  data.d  / dcomplex(other)); break;
            case SCOMPLEX: data.c =           T(data.c) /          other  ; break;
            case DCOMPLEX: data.z =             data.z  / dcomplex(other) ; break;
        }

        return *this;
    }

    template <typename T>
    std::enable_if_t<!(type_tag<T>::value & SCOMPLEX), tblis_scalar&>
    operator/=(T other)
    {
        switch (type)
        {
            case    FLOAT: data.s /= other; break;
            case   DOUBLE: data.d /= other; break;
            case SCOMPLEX: data.c = std::complex<T>(data.c) /        other ; break;
            case DCOMPLEX: data.z =                 data.z  / double(other); break;
        }

        return *this;
    }

    tblis_scalar& operator+=(const tblis_scalar& other)
    {
        if (other.type & DOUBLE) *this += other.data.z;
        else                     *this += other.data.c;

        return *this;
    }

    tblis_scalar& operator-=(const tblis_scalar& other)
    {
        if (other.type & DOUBLE) *this -= other.data.z;
        else                     *this -= other.data.c;

        return *this;
    }

    tblis_scalar& operator*=(const tblis_scalar& other)
    {
        switch (other.type)
        {
            case FLOAT:    *this *= other.data.s; break;
            case DOUBLE:   *this *= other.data.d; break;
            case SCOMPLEX: *this *= other.data.c; break;
            case DCOMPLEX: *this *= other.data.z; break;
        }

        return *this;
    }

    tblis_scalar& operator/=(const tblis_scalar& other)
    {
        switch (other.type)
        {
            case FLOAT:    *this /= other.data.s; break;
            case DOUBLE:   *this /= other.data.d; break;
            case SCOMPLEX: *this /= other.data.c; break;
            case DCOMPLEX: *this /= other.data.z; break;
        }

        return *this;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    tblis_scalar operator+(T other) const
    {
        tblis_scalar ret(*this);
        ret += other;
        return ret;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    friend tblis_scalar operator+(T lhs, const tblis_scalar& rhs)
    {
        return rhs + lhs;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    tblis_scalar operator-(T other) const
    {
        tblis_scalar ret(*this);
        ret -= other;
        return ret;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    friend tblis_scalar operator-(T lhs, const tblis_scalar& rhs)
    {
        tblis_scalar ret(rhs);
        ret.negate();
        ret += lhs;
        return ret;
    }

    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    tblis_scalar operator*(T other) const
    {
        tblis_scalar ret(*this);
        ret *= other;
        return ret;
    }


    template <typename T, typename=std::enable_if_t<is_scalar<T>::value>>
    friend tblis_scalar operator*(T lhs, const tblis_scalar& rhs)
    {
        return rhs * lhs;
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

    tblis_scalar& negate()
    {
        if (type & DOUBLE) data.z = -data.z;
        else               data.c = -data.c;
        return *this;
    }

    tblis_scalar operator-() const
    {
        tblis_scalar ret(*this);
        ret.negate();
        return ret;
    }

    tblis_scalar& conj()
    {
        switch (type)
        {
            case FLOAT:    break;
            case DOUBLE:   break;
            case SCOMPLEX: data.c = std::conj(data.c); break;
            case DCOMPLEX: data.z = std::conj(data.z); break;
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
            case FLOAT:    data.s = std::abs(data.s); break;
            case DOUBLE:   data.d = std::abs(data.d); break;
            case SCOMPLEX: data.c = std::abs(data.c); break;
            case DCOMPLEX: data.z = std::abs(data.z); break;
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
            case FLOAT:    data.s = std::sqrt(data.s); break;
            case DOUBLE:   data.d = std::sqrt(data.d); break;
            case SCOMPLEX: data.c = std::sqrt(data.c); break;
            case DCOMPLEX: data.z = std::sqrt(data.z); break;
        }

        return *this;
    }

    friend tblis_scalar sqrt(const tblis_scalar& other)
    {
        tblis_scalar ret(other);
        ret.sqrt();
        return other;
    }

#endif //TBLIS_ENABLE_CXX

} tblis_scalar;

#if TBLIS_ENABLE_CXX

using scalar = tblis_scalar;

template <> inline
float tblis_scalar::get<float>() const
{
    if (type & DOUBLE) return data.d;
    else               return data.s;
}

template <> inline
double tblis_scalar::get<double>() const
{
    if (type & DOUBLE) return data.d;
    else               return data.s;
}

template <> inline
scomplex tblis_scalar::get<scomplex>() const
{
    if (type & DOUBLE) return scomplex(data.z);
    else               return          data.c;
}

template <> inline
dcomplex tblis_scalar::get<dcomplex>() const
{
    if (type & DOUBLE) return          data.z;
    else               return dcomplex(data.c);
}

#endif //TBLIS_ENABLE_CXX

TBLIS_EXPORT void tblis_init_scalar_s(tblis_scalar* s, float value);

TBLIS_EXPORT void tblis_init_scalar_d(tblis_scalar* s, double value);

TBLIS_EXPORT void tblis_init_scalar_c(tblis_scalar* s, scomplex value);

TBLIS_EXPORT void tblis_init_scalar_z(tblis_scalar* s, dcomplex value);

namespace detail
{

template <typename T>
struct buffer
{
    std::array<T,TBLIS_OPT_NDIM> static_buf;
    std::vector<T> dyn_buf;

    template <typename U>
    buffer& operator=(const U& other)
    {
        auto it = other.begin();
        auto end = other.end();

        for (int i = 0;it != end && i < static_buf.size();++i, ++it)
            static_buf[i] = *it;

        if (it != end)
        {
            dyn_buf.assign(static_buf.begin(), static_buf.end());
            dyn_buf.insert(dyn_buf.end(), it, end);
        }

        return *this;
    }

    void resize(size_t size)
    {
        if (size <= static_buf.size())
            dyn_buf.clear();
        else
            dyn_buf.resize(size);
    }

    T* data() const
    {
        return const_cast<T*>(dyn_buf.empty() ? static_buf.data() : dyn_buf.data());
    }
};

}

#if TBLIS_ENABLE_CXX
struct const_tensor;
#endif

typedef struct tblis_tensor
{
    tblis_scalar scalar;
    int conj;
    void* data;
    int ndim;
    len_type* len;
    stride_type* stride;

#if TBLIS_ENABLE_CXX

    const type_t& type = scalar.type;
    detail::buffer<len_type> len_buf;
    detail::buffer<stride_type> stride_buf;

    template <typename T, typename=void> struct convert {};

    template <typename T, typename=void> struct convertible_helper {};

    template <typename T, bool Mutable> struct convertible_helper2
    {
        typedef T const_type;
    };

    template <typename T> struct convertible_helper2<T, true>
    {
        typedef T const_type;
        typedef T type;
    };

    template <typename T>
    struct convertible_helper<T,
        decltype(std::declval<convert<T>>()(std::declval<tblis_tensor&>(),
                                            std::declval<std::remove_const_t<T>&>()))>
    : convertible_helper2<T, convert<T>::is_mutable> {};

    template <typename T> using const_convertible =
        typename convertible_helper<std::remove_reference_t<T>>::const_type;

    template <typename T> using convertible =
        typename convertible_helper<std::remove_reference_t<T>>::type;

    template <typename T, typename=convertible<T>>
    tblis_tensor(T&& other)
    : tblis_tensor()
    {
        convert<std::remove_reference_t<T>>{}(*this, other);
    }

    //tblis_tensor(const const_tensor& other);

    //tblis_tensor(const_tensor&& other);

    tblis_tensor(const tblis_tensor& other)
    : scalar(other.scalar),
      conj(other.conj),
      data(other.data),
      ndim(other.ndim)
    {
        if (other.len == other.len_buf.data())
        {
            len_buf = other.len_buf;
            stride_buf = other.stride_buf;
            len = len_buf.data();
            stride = stride_buf.data();
        }
        else
        {
            len = other.len;
            stride = other.stride;
        }
    }

    tblis_tensor(tblis_tensor&& other)
    : scalar(other.scalar),
      conj(other.conj),
      data(other.data),
      ndim(other.ndim)
    {
        if (other.len == other.len_buf.data())
        {
            len_buf = std::move(other.len_buf);
            stride_buf = std::move(other.stride_buf);
            len = len_buf.data();
            stride = stride_buf.data();
        }
        else
        {
            len = other.len;
            stride = other.stride;
        }
    }

    tblis_tensor()
    : scalar(1.0),
      conj(false),
      data(0),
      ndim(0),
      len(0),
      stride(0)
    {}

    template <typename T>
    tblis_tensor(T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : scalar(T(1)),
      conj(false),
      data(A),
      ndim(ndim),
      len(const_cast<len_type*>(len)),
      stride(const_cast<stride_type*>(stride))
    {}

    template <typename T>
    tblis_tensor(T alpha, T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : scalar(alpha),
      conj(false),
      data(A),
      ndim(ndim),
      len(const_cast<len_type*>(len)),
      stride(const_cast<stride_type*>(stride))
    {}

    template <typename T>
    tblis_tensor(T alpha, bool conj, T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : scalar(alpha),
      conj(conj),
      data(A),
      ndim(ndim),
      len(const_cast<len_type*>(len)),
      stride(const_cast<stride_type*>(stride))
    {}

    tblis_tensor& operator=(const tblis_tensor&) = delete;

    tblis_tensor& operator=(tblis_tensor&&) = delete;

#endif //TBLIS_ENABLE_CXX

} tblis_tensor;

#if TBLIS_ENABLE_CXX

using tensor = tblis_tensor;

struct const_tensor
{
    tensor tensor_;
    const type_t& type = tensor_.type;

    template <typename T, typename=tensor::const_convertible<const T>>
    const_tensor(const T& other)
    {
        tensor::convert<T>{}(tensor_, const_cast<T&>(other));
    }

    const_tensor()
    : tensor_()
    {}

    const_tensor(const tensor& other)
    : tensor_(other)
    {}

    const_tensor(tensor&& other)
    : tensor_(std::move(other))
    {}

    template <typename T>
    const_tensor(const T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : tensor_(const_cast<T*>(A), ndim, len, stride)
    {}

    template <typename T>
    const_tensor(T alpha, const T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : tensor_(alpha, const_cast<T*>(A), ndim, len, stride)
    {}

    template <typename T>
    const_tensor(T alpha, bool conj, const T* A, int ndim,
                 const len_type* len, const stride_type* stride)
    : tensor_(alpha, conj, const_cast<T*>(A), ndim, len, stride)
    {}
};

//inline tblis_tensor::tblis_tensor(const const_tensor& other)
//: tblis_tensor(other.tensor_) {}

//inline tblis_tensor::tblis_tensor(const_tensor&& other)
//: tblis_tensor(std::move(other.tensor_)) {}

#endif //TBLIS_ENABLE_CXX

TBLIS_EXPORT void tblis_init_tensor_scaled_s(tblis_tensor* type_t, float scalar,
                                             int ndim, len_type* len, float* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_scaled_d(tblis_tensor* type_t, double scalar,
                                             int ndim, len_type* len, double* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_scaled_c(tblis_tensor* type_t, scomplex scalar,
                                             int ndim, len_type* len, scomplex* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_scaled_z(tblis_tensor* type_t, dcomplex scalar,
                                             int ndim, len_type* len, dcomplex* data,
                                             stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_s(tblis_tensor* type_t,
                                      int ndim, len_type* len, float* data,
                                      stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_d(tblis_tensor* type_t,
                                      int ndim, len_type* len, double* data,
                                      stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_c(tblis_tensor* type_t,
                                      int ndim, len_type* len, scomplex* data,
                                      stride_type* stride);

TBLIS_EXPORT void tblis_init_tensor_z(tblis_tensor* type_t,
                                      int ndim, len_type* len, dcomplex* data,
                                      stride_type* stride);

#if TBLIS_ENABLE_CXX

namespace detail
{

template <typename T, typename=void>
struct has_label_data : std::false_type {};

template <typename T>
struct has_label_data<T,
    std::enable_if_t<
        std::is_same<decltype(std::declval<std::add_const_t<T>>().data()),
                     const label_type*>::value>>
: std::true_type {};

}

struct label_string
{
    detail::buffer<label_type> idx_buf;
    label_type* idx;

    label_string() : idx(nullptr) {}

    label_string(const label_string& other)
    {
        if (other.idx == other.idx_buf.data())
        {
            idx_buf = other.idx_buf;
            idx = idx_buf.data();
        }
        else
        {
            idx = other.idx;
        }
    }

    label_string(label_string&& other)
    {
        if (other.idx == other.idx_buf.data())
        {
            idx_buf = std::move(other.idx_buf);
            idx = idx_buf.data();
        }
        else
        {
            idx = other.idx;
        }
    }

    label_string(std::initializer_list<label_type> ilist)
    {
        idx = const_cast<label_type*>(ilist.begin());
    }

    label_string(const label_type* s)
    {
        idx = const_cast<label_type*>(s);
    }

    template <typename T>
    label_string(const T& s, std::enable_if_t<detail::has_label_data<T>::value>* = 0)
    {
        idx = const_cast<label_type*>(s.data());
    }

    template <typename T>
    label_string(const T& s, std::enable_if_t<!detail::has_label_data<T>::value>* = 0)
    {
        idx_buf = s;
        idx = idx_buf.data();
    }

    label_string& operator=(const label_string&) = delete;

    label_string& operator=(label_string&&) = delete;
};

label_string idx(const const_tensor& t)
{
    label_string s;

    if (t.tensor_.ndim >= TBLIS_OPT_NDIM)
        s.idx_buf.resize(t.tensor_.ndim);
    s.idx = s.idx_buf.data();

    for (int i = 0;i < t.tensor_.ndim;i++)
        s.idx[i] = i;

    return s;
}

#ifdef MARRAY_MARRAY_BASE_HPP

template <typename T, int N, int I, typename... D>
struct tensor::convert<MArray::marray_slice<T,N,I,D...>>
{
    constexpr static bool is_mutable = !std::is_const<T>::value;

    void operator()(tensor& t, const MArray::marray_slice<T,N,I,D...>& other) const
    {
        auto other_view = other.view();
        tensor::convert<decltype(other_view)>(t, other_view);
    }
};

template <typename T, int N, typename D, bool O>
struct tensor::convert<MArray::marray_base<T,N,D,O>>
{
    constexpr static bool is_mutable = O || !std::is_const<T>::value;

    void operator()(tensor& t, MArray::marray_base<T,N,D,O>& other) const
    {
        t.scalar.reset(T(1));
        t.data = const_cast<void*>(static_cast<const void*>(other.data()));
        t.ndim = other.dimension();
        t.len_buf = other.lengths();
        t.stride_buf = other.strides();
        t.len = t.len_buf.data();
        t.stride = t.stride_buf.data();
    }
};


template <typename T, int N, typename D, bool O>
struct tensor::convert<const MArray::marray_base<T,N,D,O>> : tensor::convert<MArray::marray_base<T,N,D,O>>
{
    constexpr static bool is_mutable = !O && !std::is_const<T>::value;
};
#endif //MARRAY_MARRAY_BASE_HPP

#ifdef EIGEN_CXX11_TENSOR_TENSOR_H

namespace detail
{

template <typename T>
struct is_eigen_tensor : std::false_type {};

template <typename T, int N, int O, typename I>
struct is_eigen_tensor<Eigen::Tensor<T,N,O,I>> : std::true_type {};

template <typename T, typename D, int O, typename I>
struct is_eigen_tensor<Eigen::TensorFixedSize<T,D,O,I>> : std::true_type {};

template <typename Tensor, int O, template <class> class MP>
struct is_eigen_tensor<Eigen::TensorMap<Tensor,O,MP>> : std::true_type {};

template <typename T, typename=void>
struct eigen_expr_parser;

template <typename T>
struct eigen_expr_parser<T, std::enable_if_t<is_eigen_tensor<T>::value>>
{
    eigen_expr_parser(tensor& t, T& other)
    {
        using U = typename T::Scalar;

        t.scalar.reset(T(1));
        t.data = other.data();
        t.ndim = other.NumDimensions;
        t.len_buf = other.dimensions().begin(), other.dimensions().end();
        t.stride_buf.resize(t.ndim);
        t.len = t.len_buf.data();
        t.stride = t.stride_buf.data();

        if (other.Options & Eigen::RowMajor)
        {
            stride_type stride = 1;
            for (int i = t.ndim-1;i >= 0;i--)
            {
                t.stride[i] = stride;
                stride *= t.len[i];
            }
        }
        else
        {
            stride_type stride = 1;
            for (int i = 0;i < t.ndim;i++)
            {
                t.stride[i] = stride;
                stride *= t.len[i];
            }
        }
    }
};

template <typename I, typename S, typename X>
struct eigen_expr_parser<Eigen::TensorSlicingOp<I,S,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorSlicingOp<I,S,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        for (int i = 0;i < t.ndim;i++)
        {
            t.data += t.stride[i] * other.startIndices()[i];
            t.len[i] = other.sizes()[i];
        }
    }
};

template <Eigen::DenseIndex D, typename X>
struct eigen_expr_parser<Eigen::TensorChippingOp<D,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorChippingOp<D,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        t.data += t.stride[other.dim()] * other.offset();

        for (int i = other.dim()+1;i < t.ndim;i++)
        {
            t.len[i] = t.len[i+1];
            t.stride[i] = t.stride[i+1];
        }

        t.ndim--;
    }
};

template <typename S, typename X>
struct eigen_expr_parser<Eigen::TensorShufflingOp<S,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorShufflingOp<S,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        auto len_buf2(t.len_buf);
        auto stride_buf2(t.stride_buf);
        auto len2 = len_buf2.data();
        auto stride2 = stride_buf2.data();

        for (int i = 0;i < t.ndim;i++)
        {
            auto perm = other.shufflePermutation()[i];
            len2[i] = t.len[perm];
            stride2[i] = t.stride[perm];
        }

        t.len_buf = len2;
        t.stride_buf = stride2;
        t.len = t.len_buf.data();
        t.stride = t.stride_buf.data();
    }
};

template <typename S, typename X>
struct eigen_expr_parser<Eigen::TensorStridingOp<S,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorStridingOp<S,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        for (int i = 0;i < t.ndim;i++)
        {
            auto stride = other.strides()[i];
            t.stride[i] *= stride;
            t.len[i] = (t.len[i] + stride - 1) / stride;
        }
    }
};

/*
template <typename Start, typename Stop, typename Stride, typename X>
struct eigen_expr_parser<Eigen::TensorStridingSlicingOp<Start,Stop,Stride,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorStridingSlicingOp<Start,Stop,Stride,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        //TODO
    }
};
*/

template <typename R, typename X>
struct eigen_expr_parser<Eigen::TensorReverseOp<R,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorReverseOp<R,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        for (int i = 0;i < t.ndim;i++)
        {
            if (other.reverse()[i])
            {
                t.data += (t.len[i] - 1) * t.stride[i];
                t.stride[i] = -t.stride[i];
            }
        }
    }
};

template <typename T, typename X>
struct eigen_expr_parser<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<T>,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<T>,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        t.scalar.negate();
    }
};

template <typename T, typename X>
struct eigen_expr_parser<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_conjugate_op<T>,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<T>,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        t.scalar.conj();
        t.conj = !t.conj;
    }
};

template <typename T, typename X>
struct eigen_expr_parser<Eigen::TensorCwiseUnaryOp<Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<T,T>>,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<T,T>>,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        t.scalar *= other.functor().m_value;
    }
};

template <typename T, typename X>
struct eigen_expr_parser<Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<T,T>>,X>> : eigen_expr_parser<X>
{
    eigen_expr_parser(tensor& t, Eigen::TensorCwiseUnaryOp<Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<T,T>>,X>& other)
    : eigen_expr_parser<X>(t, other.expression())
    {
        t.scalar *= other.functor().m_value;
    }
};

}

template <typename D>
struct tensor::convert<D, std::enable_if_t<std::is_base_of<Eigen::TensorBase<D,Eigen::ReadOnlyAccessors>,D>::value>>
{
    constexpr static bool is_mutable = false;

    void operator()(tensor& t, Eigen::TensorBase<D,Eigen::ReadOnlyAccessors>& other) const
    {
        detail::eigen_expr_parser<D>{}(static_cast<D&>(t), other);
    }
};

#endif //EIGEN_CXX11_TENSOR_TENSOR_MAP_H

#endif //TBLIS_ENABLE_CXX

TBLIS_END_NAMESPACE

#endif //TBLIS_BASE_TYPES_H
