#ifndef _TENSOR_BLIS___HPP_
#define _TENSOR_BLIS___HPP_

#include "blis.h"
#include "bli_obj_macro_defs.h"
#include "bli_scalar_macro_defs.h"

#include <complex>
#include <type_traits>

namespace blis
{

template <typename T> struct real_type                  { typedef T type; };
template <typename T> struct real_type<std::complex<T>> { typedef T type; };
template <typename T> using real_type_t = typename real_type<T>::type;

template <typename T> struct complex_type                  { typedef std::complex<T> type; };
template <typename T> struct complex_type<std::complex<T>> { typedef std::complex<T> type; };
template <typename T> using complex_type_t = typename complex_type<T>::type;

template <typename T> struct is_real                  { static const bool value =  true; };
template <typename T> struct is_real<std::complex<T>> { static const bool value = false; };

template <typename T> struct is_complex                  { static const bool value = false; };
template <typename T> struct is_complex<std::complex<T>> { static const bool value =  true; };

namespace detail
{
    template <typename T, typename U,
              typename V = std::complex<typename std::common_type<real_type_t<T>,
                                                                  real_type_t<U>>::type>>
    using complex_op =
        typename std::enable_if<is_complex<T>::value ||
                                is_complex<U>::value,V>::type;
}

typedef std::complex< float> sComplex;
typedef std::complex<double> dComplex;

inline sComplex cmplx(const scomplex& x) { return sComplex(bli_creal(x), bli_cimag(x)); }

inline dComplex cmplx(const dcomplex& x) { return dComplex(bli_creal(x), bli_cimag(x)); }

template <typename T, typename U>
detail::complex_op<T,U> operator+(const T& a, const U& b)
{
    typedef detail::complex_op<T,U> V;
    return V(a) + V(b);
}

template <typename T, typename U>
detail::complex_op<T,U> operator-(const T& a, const U& b)
{
    typedef detail::complex_op<T,U> V;
    return V(a) - V(b);
}

template <typename T, typename U>
detail::complex_op<T,U> operator*(const T& a, const U& b)
{
    typedef detail::complex_op<T,U> V;
    return V(a) * V(b);
}

template <typename T, typename U>
detail::complex_op<T,U> operator/(const T& a, const U& b)
{
    typedef detail::complex_op<T,U> V;
    return V(a) / V(b);
}

template <typename T> struct datatype;
template <> struct datatype<   float> { static const num_t value =    BLIS_FLOAT; };
template <> struct datatype<  double> { static const num_t value =   BLIS_DOUBLE; };
template <> struct datatype<scomplex> { static const num_t value = BLIS_SCOMPLEX; };
template <> struct datatype<dcomplex> { static const num_t value = BLIS_DCOMPLEX; };
template <> struct datatype<sComplex> { static const num_t value = BLIS_SCOMPLEX; };
template <> struct datatype<dComplex> { static const num_t value = BLIS_DCOMPLEX; };

template <typename T>
typename std::enable_if<!is_complex<T>::value,T>::type
conj(T val) { return val; }

template <typename T>
typename std::enable_if<is_complex<T>::value,T>::type
conj(T val) { return std::conj(val); }

using std::real;
using std::imag;

template <typename T>
typename std::enable_if<!is_complex<T>::value,T>::type
norm2(T val) { return val*val; }

template <typename T>
typename std::enable_if<is_complex<T>::value,T>::type
norm2(T val) { return real(val)*real(val) + imag(val)*imag(val); }

template <typename T, typename U>
detail::complex_op<T,U,bool> operator==(const T& a, const U& b)
{
    return real(a) == real(b) && imag(a) == imag(b);
}

template <typename T, typename U>
detail::complex_op<T,U,bool> operator!=(const T& a, const U& b)
{
    return !(a == b);
}

template <typename T, typename U>
detail::complex_op<T,U,bool> operator<(const T& a, const U& b)
{
    return real(a)+imag(a) < real(b)+imag(b);
}

template <typename T, typename U>
detail::complex_op<T,U,bool> operator>(const T& a, const U& b)
{
    return b < a;
}

template <typename T, typename U>
detail::complex_op<T,U,bool> operator<=(const T& a, const U& b)
{
    return !(b < a);
}

template <typename T, typename U>
detail::complex_op<T,U,bool> operator>=(const T& a, const U& b)
{
    return !(a < b);
}

template <typename T> struct MC {};
template <> struct MC<   float> { static constexpr dim_t value = BLIS_DEFAULT_MC_S; };
template <> struct MC<  double> { static constexpr dim_t value = BLIS_DEFAULT_MC_D; };
template <> struct MC<sComplex> { static constexpr dim_t value = BLIS_DEFAULT_MC_C; };
template <> struct MC<dComplex> { static constexpr dim_t value = BLIS_DEFAULT_MC_Z; };

template <typename T> struct NC {};
template <> struct NC<   float> { static constexpr dim_t value = BLIS_DEFAULT_NC_S; };
template <> struct NC<  double> { static constexpr dim_t value = BLIS_DEFAULT_NC_D; };
template <> struct NC<sComplex> { static constexpr dim_t value = BLIS_DEFAULT_NC_C; };
template <> struct NC<dComplex> { static constexpr dim_t value = BLIS_DEFAULT_NC_Z; };

template <typename T> struct KC {};
template <> struct KC<   float> { static constexpr dim_t value = BLIS_DEFAULT_KC_S; };
template <> struct KC<  double> { static constexpr dim_t value = BLIS_DEFAULT_KC_D; };
template <> struct KC<sComplex> { static constexpr dim_t value = BLIS_DEFAULT_KC_C; };
template <> struct KC<dComplex> { static constexpr dim_t value = BLIS_DEFAULT_KC_Z; };

template <typename T> struct MR {};
template <> struct MR<   float> { static constexpr dim_t value = BLIS_DEFAULT_MR_S; };
template <> struct MR<  double> { static constexpr dim_t value = BLIS_DEFAULT_MR_D; };
template <> struct MR<sComplex> { static constexpr dim_t value = BLIS_DEFAULT_MR_C; };
template <> struct MR<dComplex> { static constexpr dim_t value = BLIS_DEFAULT_MR_Z; };

template <typename T> struct NR {};
template <> struct NR<   float> { static constexpr dim_t value = BLIS_DEFAULT_NR_S; };
template <> struct NR<  double> { static constexpr dim_t value = BLIS_DEFAULT_NR_D; };
template <> struct NR<sComplex> { static constexpr dim_t value = BLIS_DEFAULT_NR_C; };
template <> struct NR<dComplex> { static constexpr dim_t value = BLIS_DEFAULT_NR_Z; };

template <typename T> struct KR {};
template <> struct KR<   float> { static constexpr dim_t value = BLIS_DEFAULT_KR_S; };
template <> struct KR<  double> { static constexpr dim_t value = BLIS_DEFAULT_KR_D; };
template <> struct KR<sComplex> { static constexpr dim_t value = BLIS_DEFAULT_KR_C; };
template <> struct KR<dComplex> { static constexpr dim_t value = BLIS_DEFAULT_KR_Z; };

template <typename T> struct gemm_ukr_t {};
template <> struct gemm_ukr_t<   float> { static constexpr sgemm_ukr_t value = BLIS_SGEMM_UKERNEL; };
template <> struct gemm_ukr_t<  double> { static constexpr dgemm_ukr_t value = BLIS_DGEMM_UKERNEL; };
template <> struct gemm_ukr_t<sComplex> { static constexpr cgemm_ukr_t value = BLIS_CGEMM_UKERNEL; };
template <> struct gemm_ukr_t<dComplex> { static constexpr zgemm_ukr_t value = BLIS_ZGEMM_UKERNEL; };

template <typename T> struct basic_type { typedef T type; };
template <> struct basic_type<sComplex> { typedef scomplex type; };
template <> struct basic_type<dComplex> { typedef dcomplex type; };
template <typename T> using basic_type_t = typename basic_type<T>::type;

}

#endif
