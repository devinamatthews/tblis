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

#ifndef TBLIS_ENABLE_CPLUSPLUS

#if defined(__cplusplus) && !defined(TBLIS_DISABLE_CPLUSPLUS)
#if __cplusplus >= 202002L
#define TBLIS_ENABLE_CPLUSPLUS 1
#else
#define TBLIS_ENABLE_CPLUSPLUS 0
#warning TBLIS: C++ interface disabled because C++20 is required
#endif
#else
#define TBLIS_ENABLE_CPLUSPLUS 0
#endif

#endif //TBLIS_ENABLE_CPLUSPLUS

#include "tblis/tblis_config.h"

#define TBLIS_STRINGIZE_(...) #__VA_ARGS__
#define TBLIS_STRINGIZE(...) TBLIS_STRINGIZE_(__VA_ARGS__)
#define TBLIS_CONCAT_(x,y) x##y
#define TBLIS_CONCAT(x,y) TBLIS_CONCAT_(x,y)
#define TBLIS_FIRST_ARG(arg,...) arg

#ifdef __cplusplus
#include <complex>
typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;
#else
typedef _Complex float scomplex;
typedef _Complex double dcomplex;
#endif

#if TBLIS_ENABLE_CPLUSPLUS

    extern "C"
    {

        typedef struct obj_s obj_t;
        typedef struct auxinfo_s auxinfo_t;
        typedef struct cntx_s cntx_t;
        typedef struct cntl_s cntl_t;
        typedef struct thrinfo_s thrinfo_t;

    }

    #include <utility>
    #include <array>

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

    #define TBLIS_EXPORT extern "C"
    #define TBLIS_BEGIN_NAMESPACE namespace tblis {
    #define TBLIS_END_NAMESPACE }

#else //TBLIS_ENABLE_CPLUSPLUS

    #define TBLIS_EXPORT
    #define TBLIS_BEGIN_NAMESPACE
    #define TBLIS_END_NAMESPACE

#endif //TBLIS_ENABLE_CPLUSPLUS

TBLIS_BEGIN_NAMESPACE

    typedef void tblis_config;

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

    /*
     * Note: these are hard-coded from blis.h to avoid bringing
     * in the whole header as a dependency.
     */
    typedef int type_t;
    static const type_t TYPE_SINGLE   = 0;
    static const type_t TYPE_FLOAT    = TYPE_SINGLE;
    static const type_t TYPE_DOUBLE   = 2;
    static const type_t TYPE_SCOMPLEX = 1;
    static const type_t TYPE_DCOMPLEX = 3;

    typedef TBLIS_LEN_TYPE len_type;
    typedef TBLIS_STRIDE_TYPE stride_type;
    typedef TBLIS_LABEL_TYPE label_type;
    #define TBLIS_MAX_UNROLL 8

    #if TBLIS_ENABLE_CPLUSPLUS

        using scomplex = std::complex<float>;
        using dcomplex = std::complex<double>;

        using std::complex;
        using std::real;
        using std::imag;

        // The following is from stl_ext
        // In the future stl_ext will be removed but these parts are still needed
        // Tell stl_ext to not define them itself for now.
        #define _STL_EXT_COMPLEX_HPP_

        template <typename T> struct real_type             { typedef T type; };
        template <typename T> struct real_type<complex<T>> { typedef T type; };
        template <typename T>
        using real_type_t = typename real_type<T>::type;

        template <typename T> struct complex_type             { typedef complex<T> type; };
        template <typename T> struct complex_type<complex<T>> { typedef complex<T> type; };
        template <typename T>
        using complex_type_t = typename complex_type<T>::type;

        template <typename T> struct is_complex             : std::false_type {};
        template <typename T> struct is_complex<complex<T>> :  std::true_type {};
        template <typename T> constexpr static auto is_complex_v = is_complex<T>::value;

        template <typename T>
        std::enable_if_t<is_complex_v<T>,T> conj(T x)
        {
            return {x.real(), -x.imag()};
        }

        template <typename T>
        std::enable_if_t<std::is_arithmetic_v<T>,T> conj(T x)
        {
            return x;
        }

        template <typename T>
        T conj(bool conjugate, T val)
        {
            return (conjugate ? conj(val) : val);
        }

        template <typename T>
        std::enable_if_t<is_complex_v<T>,real_type_t<T>> norm2(T x)
        {
            return x.real()*x.real() + x.imag()*x.imag();
        }

        template <typename T>
        std::enable_if_t<std::is_arithmetic_v<T>,T> norm2(T x)
        {
            return x*x;
        }

    #endif //TBLIS_ENABLE_CPLUSPLUS

TBLIS_END_NAMESPACE

#if TBLIS_ENABLE_CPLUSPLUS

    namespace std
    {

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator+(const complex<T>& f, const std::complex<U>& d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)+complex<V>(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator+(const complex<T>& f, U d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)+V(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator+(T d, const complex<U>& f)
        {
            typedef std::common_type_t<T,U> V;
            return V(d)+complex<V>(f);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator-(const complex<T>& f, const std::complex<U>& d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)-complex<V>(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator-(const complex<T>& f, U d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)-V(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator-(T d, const complex<U>& f)
        {
            typedef std::common_type_t<T,U> V;
            return V(d)-complex<V>(f);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator*(const complex<T>& f, const std::complex<U>& d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)*complex<V>(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator*(const complex<T>& f, U d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)*V(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator*(T d, const complex<U>& f)
        {
            typedef std::common_type_t<T,U> V;
            return V(d)*complex<V>(f);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator/(const complex<T>& f, const std::complex<U>& d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)/complex<V>(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator/(const complex<T>& f, U d)
        {
            typedef std::common_type_t<T,U> V;
            return complex<V>(f)/V(d);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                    std::is_arithmetic_v<U> &&
                    !std::is_same<T,U>::value,complex<std::common_type_t<T,U>>>
        operator/(T d, const complex<U>& f)
        {
            typedef std::common_type_t<T,U> V;
            return V(d)/complex<V>(f);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator<(const complex<T>& a, const complex<U>& b)
        {
            return a.real() < b.real();
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator>(const complex<T>& a, const complex<U>& b)
        {
            return b < a;
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator<=(const complex<T>& a, const complex<U>& b)
        {
            return !(b < a);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator>=(const complex<T>& a, const complex<U>& b)
        {
            return !(a < b);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator<(const complex<T>& a, U b)
        {
            return a.real() < b;
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator>(const complex<T>& a, U b)
        {
            return b < a;
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator<=(const complex<T>& a, U b)
        {
            return !(b < a);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator>=(const complex<T>& a, U b)
        {
            return !(a < b);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator<(T a, const complex<U>& b)
        {
            return a < b.real();
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator>(T a, const complex<U>& b)
        {
            return b < a;
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator<=(T a, const complex<U>& b)
        {
            return !(b < a);
        }

        template <typename T, typename U>
        std::enable_if_t<std::is_arithmetic_v<T> &&
                             std::is_arithmetic_v<U>,bool>
        operator>=(T a, const complex<U>& b)
        {
            return !(a < b);
        }

    }

    TBLIS_BEGIN_NAMESPACE

        template <typename T> struct type_tag { static constexpr type_t value =   TYPE_DOUBLE; };
        template <> struct type_tag<   float> { static constexpr type_t value =    TYPE_FLOAT; };
        template <> struct type_tag<  double> { static constexpr type_t value =   TYPE_DOUBLE; };
        template <> struct type_tag<scomplex> { static constexpr type_t value = TYPE_SCOMPLEX; };
        template <> struct type_tag<dcomplex> { static constexpr type_t value = TYPE_DCOMPLEX; };

        constexpr static std::array<size_t,4> type_size =
        {
            sizeof(   float),
            sizeof(  double),
            sizeof(scomplex),
            sizeof(dcomplex),
        };

        constexpr static std::array<size_t,4> type_alignment =
        {
            alignof(   float),
            alignof(  double),
            alignof(scomplex),
            alignof(dcomplex),
        };

        namespace matrix_constants
        {
            enum {MAT_A, MAT_B, MAT_C};
            enum {DIM_M, DIM_N, DIM_K};
        }

        #define DO_FOREACH_TYPE \
        FOREACH_TYPE(float); \
        FOREACH_TYPE(double); \
        FOREACH_TYPE(scomplex); \
        FOREACH_TYPE(dcomplex);

    TBLIS_END_NAMESPACE

#endif //TBLIS_ENABLE_CPLUSPLUS

TBLIS_BEGIN_NAMESPACE

    typedef struct tblis_scalar
    {
        union scalar
        {
            float s;
            double d;
            scomplex c;
            dcomplex z;

            #if TBLIS_ENABLE_CPLUSPLUS

                scalar() : z(0.0, 0.0) {}

            #endif
        } data;
        type_t type;

        #if TBLIS_ENABLE_CPLUSPLUS

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
            tblis_scalar(T value)
            : type(type_tag<T>::value)
            {
                *this = value;
            }

            template <typename T>
            tblis_scalar(T value, type_t type)
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

            template <typename T>
            T as() const
            {
                if constexpr (is_complex_v<T>)
                {
                    switch (type)
                    {
                        case TYPE_FLOAT:    return T(data.s); break;
                        case TYPE_DOUBLE:   return T(data.d); break;
                        case TYPE_SCOMPLEX: return T(data.c); break;
                        case TYPE_DCOMPLEX: return T(data.z); break;
                        default: break;
                    }
                }
                else
                {
                    switch (type)
                    {
                        case TYPE_FLOAT:    return T(data.s); break;
                        case TYPE_DOUBLE:   return T(data.d); break;
                        case TYPE_SCOMPLEX: return T(data.c.real()); break;
                        case TYPE_DCOMPLEX: return T(data.z.real()); break;
                        default: break;
                    }
                }

                return T{};
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
                            case TYPE_FLOAT:    data.s = float(          other.data.s ); break;
                            case TYPE_DOUBLE:   data.s = float(          other.data.d ); break;
                            case TYPE_SCOMPLEX: data.s = float(std::real(other.data.c)); break;
                            case TYPE_DCOMPLEX: data.s = float(std::real(other.data.z)); break;
                            default: break;
                        }
                        break;
                    case TYPE_DOUBLE:
                        switch (other.type)
                        {
                            case TYPE_FLOAT:    data.d = double(          other.data.s ); break;
                            case TYPE_DOUBLE:   data.d = double(          other.data.d ); break;
                            case TYPE_SCOMPLEX: data.d = double(std::real(other.data.c)); break;
                            case TYPE_DCOMPLEX: data.d = double(std::real(other.data.z)); break;
                            default: break;
                        }
                        break;
                    case TYPE_SCOMPLEX:
                        switch (other.type)
                        {
                            case TYPE_FLOAT:    data.c = scomplex(other.data.s); break;
                            case TYPE_DOUBLE:   data.c = scomplex(other.data.d); break;
                            case TYPE_SCOMPLEX: data.c = scomplex(other.data.c); break;
                            case TYPE_DCOMPLEX: data.c = scomplex(other.data.z); break;
                            default: break;
                        }
                        break;
                    case TYPE_DCOMPLEX:
                        switch (other.type)
                        {
                            case TYPE_FLOAT:    data.z = dcomplex(other.data.s); break;
                            case TYPE_DOUBLE:   data.z = dcomplex(other.data.d); break;
                            case TYPE_SCOMPLEX: data.z = dcomplex(other.data.c); break;
                            case TYPE_DCOMPLEX: data.z = dcomplex(other.data.z); break;
                            default: break;
                        }
                        break;
                    default:
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
                    case TYPE_FLOAT:    data.s = float   (std::real(value)); break;
                    case TYPE_DOUBLE:   data.d = double  (std::real(value)); break;
                    case TYPE_SCOMPLEX: data.c = scomplex(          value ); break;
                    case TYPE_DCOMPLEX: data.z = dcomplex(          value ); break;
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
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
                    default: break;
                }

                return *this;
            }

            friend tblis_scalar sqrt(const tblis_scalar& other)
            {
                tblis_scalar ret(other);
                ret.sqrt();
                return other;
            }

            friend std::ostream& operator<<(std::ostream& os, const tblis_scalar& x)
            {
                switch (x.type)
                {
                    case TYPE_FLOAT:    os << x.data.s; break;
                    case TYPE_DOUBLE:   os << x.data.d; break;
                    case TYPE_SCOMPLEX: os << '(' << x.data.c.real() << ',' << x.data.c.imag() << ')'; break;
                    case TYPE_DCOMPLEX: os << '(' << x.data.z.real() << ',' << x.data.z.imag() << ')'; break;
                    default: break;
                }

                return os;
            }

        #endif //TBLIS_ENABLE_CPLUSPLUS

    } tblis_scalar;

    #if TBLIS_ENABLE_CPLUSPLUS

        template <> inline
        float& tblis_scalar::get<float>() { return data.s; }

        template <> inline
        double& tblis_scalar::get<double>() { return data.d; }

        template <> inline
        scomplex& tblis_scalar::get<scomplex>() { return data.c; }

        template <> inline
        dcomplex& tblis_scalar::get<dcomplex>() { return data.z; }

    #endif //TBLIS_ENABLE_CPLUSPLUS

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

        #if TBLIS_ENABLE_CPLUSPLUS

            tblis_tensor()
            : type(TYPE_DOUBLE), conj(false), scalar(1.0), data(nullptr),
              ndim(0), len(nullptr), stride(nullptr) {}

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

        #endif //TBLIS_ENABLE_CPLUSPLUS

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

TBLIS_END_NAMESPACE

#if TBLIS_ENABLE_CPLUSPLUS

    #ifdef TBLIS_ENABLE_COMPAT

        #define MARRAY_DISABLE_VECTOR_CONSTRUCTOR
        #include "marray/marray.hpp"
        #include <memory>

        TBLIS_BEGIN_NAMESPACE

            template <typename T, typename Allocator=std::allocator<T>>
            using tensor = MArray::marray<T, MArray::DYNAMIC, Allocator>;

            template <typename T, typename Allocator=std::allocator<T>>
            using varray = MArray::marray<T, MArray::DYNAMIC, Allocator>;

            template <typename T>
            using varray_view = MArray::marray_view<T, MArray::DYNAMIC>;

            using namespace MArray;

        TBLIS_END_NAMESPACE

        #define TBLIS_COMPAT_INLINE template <typename T_=void>

    #else //TBLIS_ENABLE_COMPAT

        #define TBLIS_COMPAT_INLINE inline

    #endif //TBLIS_ENABLE_COMPAT

    #include <string>

    #include "marray/range.hpp"
    #include "marray/short_vector.hpp"
    #include "marray/index_iterator.hpp"

    TBLIS_BEGIN_NAMESPACE

        using MArray::range;

        using label_vector = std::conditional_t<
            std::is_same_v<label_type,char>,
            std::string,
            MArray::short_vector<label_type,MARRAY_OPT_NDIM>>;
        using len_vector = MArray::short_vector<len_type,MARRAY_OPT_NDIM>;
        using stride_vector = MArray::short_vector<stride_type,MARRAY_OPT_NDIM>;

        using MArray::detail::ipow;

        template <int N=1>
        using viterator = MArray::index_iterator<MArray::DYNAMIC, N>;

        using scalar = tblis_scalar;

        struct tensor_wrapper : tblis_tensor
        {
            len_vector len_buf;
            stride_vector stride_buf;

            #if defined(MARRAY_MARRAY_BASE_HPP)

                template <typename T, int N, typename D, bool O>
                tensor_wrapper(const MArray::marray_base<T,N,D,O>& t)
                : tblis_tensor(t.data(), t.dimension(),
                               t.lengths().data(), t.strides().data()) {}

                template <typename T, int N, typename D, bool O>
                tensor_wrapper(MArray::marray_base<T,N,D,O>&& t)
                : tblis_tensor(t.data(), N, nullptr, nullptr)
                {
                    len_buf.assign(t.lengths().begin(), t.lengths().end());
                    stride_buf.assign(t.strides().begin(), t.strides().end());
                    len = len_buf.data();
                    stride = stride_buf.data();
                }

            #endif

            #if defined(MARRAY_MARRAY_SLICE_HPP)

                template <typename T, int N, int I, typename... D>
                tensor_wrapper(const MArray::marray_slice<T,N,I,D...>& t) : tensor_wrapper(t.view()) {}

            #endif

            #if defined(EIGEN_CXX11_TENSOR_TENSOR_H)

                template <typename T, int N, int O, typename I>
                tensor_wrapper(const Eigen::Tensor<T,N,O,I>& t)
                : tblis_tensor(t.data(), N, nullptr, nullptr)
                {
                    auto dims = t.dimensions();
                    len_buf.assign(dims.begin(), dims.end());
                    stride_buf = marray<T>::strides(len_buf, t.Options&Eigen::RowMajor ? ROW_MAJOR : COLUMN_MAJOR);
                    len = len_buf.data();
                    stride = stride_buf.data();
                }

            #endif

            #if defined(EIGEN_CXX11_TENSOR_TENSOR_FIXED_SIZE_H)

                template <typename T, typename D, int O, typename I>
                tensor_wrapper(const Eigen::TensorFixedSize<T,D,O,I>& t)
                : tblis_tensor(t.data(), t.NumIndices, nullptr, nullptr)
                {
                    auto dims = t.dimensions();
                    len_buf.assign(dims.begin(), dims.end());
                    stride_buf = marray<T>::strides(len_buf, t.Options&Eigen::RowMajor ? ROW_MAJOR : COLUMN_MAJOR);
                    len = len_buf.data();
                    stride = stride_buf.data();
                }

            #endif

            #if defined(EIGEN_CXX11_TENSOR_TENSOR_MAP_H)

                template <typename Tensor, int O, template <class> class MP>
                tensor_wrapper(const Eigen::TensorMap<Tensor,O,MP>& t)
                : tblis_tensor(t.data(), t.NumIndices, nullptr, nullptr)
                {
                    auto dims = t.dimensions();
                    len_buf.assign(dims.begin(), dims.end());
                    stride_buf = marray<double>::strides(len_buf, Tensor::Options&Eigen::RowMajor ? ROW_MAJOR : COLUMN_MAJOR);
                    len = len_buf.data();
                    stride = stride_buf.data();
                }

            #endif

        };

        inline label_vector idx(const tblis_tensor& A, label_vector&& = label_vector())
        {
            return MArray::range(A.ndim);
        }

        label_vector idx(const std::string& from, label_vector&& to = label_vector());

    TBLIS_END_NAMESPACE

#endif //TBLIS_ENABLE_CPLUSPLUS

#endif
