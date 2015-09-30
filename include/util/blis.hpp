#ifndef _TENSOR_UTIL_BLIS_HPP_
#define _TENSOR_UTIL_BLIS_HPP_

#include "blis.h"
#include "bli_obj_macro_defs.h"
#include "bli_scalar_macro_defs.h"

#include <cmath>
#include <type_traits>
#include <iomanip>
#include <ostream>

namespace blis
{

inline float norm2(float x)
{
    return x*x;
}

inline double norm2(double x)
{
    return x*x;
}

inline float sgn(float x)
{
    return (x < 0.0f ? -1.0f : 1.0f);
}

inline double sgn(double x)
{
    return (x < 0.0 ? -1.0 : 1.0);
}

template <typename T> class Complex;

template <>
class Complex<float>
{
    friend class Complex<double>;

    private:
        scomplex val;

    public:
        Complex(float r = 0.0f, float i = 0.0f)
        {
            bli_csets(r, i, val);
        }

        Complex(double r, double i = 0.0f)
        {
            bli_csets(r, i, val);
        }

        template <class I>
        Complex(I r, I i = 0, typename std::enable_if<std::is_integral<I>::value>::type* foo = 0)
        {
            bli_csets(r, i, val);
        }

        Complex(const scomplex& x)
        {
            bli_csets(bli_creal(x), bli_cimag(x), val);
        }

        Complex(const dcomplex& x)
        {
            bli_csets(bli_zreal(x), bli_zimag(x), val);
        }

        Complex(const Complex& x)
        {
            bli_csets(x.real(), x.imag(), val);
        }

        template <typename T>
        Complex(const Complex<T>& x)
        {
            bli_csets(x.real(), x.imag(), val);
        }

        float real() const { return bli_creal(val); }

        float imag() const { return bli_cimag(val); }

        void real(float x) { bli_csets(x, bli_cimag(val), val); }

        void imag(float x) { bli_csets(bli_creal(val), x, val); }

        Complex conj() const { return Complex(real(), -imag()); }

        Complex& operator=(const scomplex& other)
        {
            bli_csets(bli_creal(other), bli_cimag(other), val);
            return *this;
        }

        Complex& operator=(const dcomplex& other)
        {
            bli_zcsets(bli_zreal(other), bli_zimag(other), val);
            return *this;
        }

        Complex& operator=(const Complex& other)
        {
            bli_csets(other.real(), other.imag(), val);
            return *this;
        }

        template <typename T>
        Complex& operator=(const Complex<T>& other)
        {
            bli_csets(other.real(), other.imag(), val);
            return *this;
        }

        Complex& operator=(float other)
        {
            bli_scsets(other, 0.0f, val);
            return *this;
        }

        Complex& operator=(double other)
        {
            bli_dcsets(other, 0.0f, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator=(I other)
        {
            bli_scsets(other, 0.0f, val);
            return *this;
        }

        Complex& operator+=(const scomplex& other)
        {
            bli_cadds(other, val);
            return *this;
        }

        Complex& operator+=(const dcomplex& other)
        {
            bli_zcadds(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator+=(const Complex<T>& other)
        {
            bli_cadds(other.val, val);
            return *this;
        }

        Complex& operator+=(float other)
        {
            bli_scadds(other, val);
            return *this;
        }

        Complex& operator+=(double other)
        {
            bli_dcadds(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator+=(I other)
        {
            bli_scadds(other, val);
            return *this;
        }

        Complex& operator-=(const scomplex& other)
        {
            bli_csubs(other, val);
            return *this;
        }

        Complex& operator-=(const dcomplex& other)
        {
            bli_zcsubs(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator-=(const Complex<T>& other)
        {
            bli_csubs(other.val, val);
            return *this;
        }

        Complex& operator-=(float other)
        {
            bli_scsubs(other, val);
            return *this;
        }

        Complex& operator-=(double other)
        {
            bli_dcsubs(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator-=(I other)
        {
            bli_scsubs(other, val);
            return *this;
        }

        Complex& operator*=(const scomplex& other)
        {
            bli_cscals(other, val);
            return *this;
        }

        Complex& operator*=(const dcomplex& other)
        {
            bli_zcscals(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator*=(const Complex<T>& other)
        {
            bli_zcscals(other.val, val);
            return *this;
        }

        Complex& operator*=(float other)
        {
            bli_scscals(other, val);
            return *this;
        }

        Complex& operator*=(double other)
        {
            bli_dcscals(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator*=(I other)
        {
            bli_scscals(other, val);
            return *this;
        }

        Complex& operator/=(const scomplex& other)
        {
            bli_cinvscals(other, val);
            return *this;
        }

        Complex& operator/=(const dcomplex& other)
        {
            bli_zcinvscals(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator/=(const Complex<T>& other)
        {
            bli_cinvscals(other.val, val);
            return *this;
        }

        Complex& operator/=(float other)
        {
            bli_szinvscals(other, val);
            return *this;
        }

        Complex& operator/=(double other)
        {
            bli_dcinvscals(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator/=(I other)
        {
            bli_scinvscals(other, val);
            return *this;
        }
};

template <>
class Complex<double>
{
    friend class Complex<float>;

    private:
        dcomplex val;

    public:
        Complex(float r, float i = 0.0f)
        {
            bli_czsets(r, i, val);
        }

        Complex(double r = 0.0, double i = 0.0f)
        {
            bli_zsets(r, i, val);
        }

        template <class I>
        Complex(I r, I i = 0, typename std::enable_if<std::is_integral<I>::value>::type* foo = 0)
        {
            bli_zsets(r, i, val);
        }

        Complex(const scomplex& x)
        {
            bli_zcsets(bli_creal(x), bli_cimag(x), val);
        }

        Complex(const dcomplex& x)
        {
            bli_zsets(bli_zreal(x), bli_zimag(x), val);
        }

        Complex(const Complex& x)
        {
            bli_zcsets(x.real(), x.imag(), val);
        }

        template <typename T>
        Complex(const Complex<T>& x)
        {
            bli_zsets(x.real(), x.imag(), val);
        }

        double real() const { return bli_zreal(val); }

        double imag() const { return bli_zimag(val); }

        void real(double x) { bli_zsets(x, bli_zimag(val), val); }

        void imag(double x) { bli_zsets(bli_zreal(val), x, val); }

        Complex conj() const { return Complex(real(), -imag()); }

        Complex& operator=(const scomplex& other)
        {
            bli_czsets(bli_creal(other), bli_cimag(other), val);
            return *this;
        }

        Complex& operator=(const dcomplex& other)
        {
            bli_zsets(bli_zreal(other), bli_zimag(other), val);
            return *this;
        }

        Complex& operator=(const Complex& other)
        {
            bli_zsets(other.real(), other.imag(), val);
            return *this;
        }

        template <typename T>
        Complex& operator=(const Complex<T>& other)
        {
            bli_zsets(other.real(), other.imag(), val);
            return *this;
        }

        Complex& operator=(float other)
        {
            bli_szsets(other, 0.0f, val);
            return *this;
        }

        Complex& operator=(double other)
        {
            bli_dzsets(other, 0.0f, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator=(I other)
        {
            bli_dzsets(other, 0.0, val);
            return *this;
        }

        Complex& operator+=(const scomplex& other)
        {
            bli_czadds(other, val);
            return *this;
        }

        Complex& operator+=(const dcomplex& other)
        {
            bli_zadds(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator+=(const Complex<T>& other)
        {
            bli_zadds(other.val, val);
            return *this;
        }

        Complex& operator+=(float other)
        {
            bli_szadds(other, val);
            return *this;
        }

        Complex& operator+=(double other)
        {
            bli_dzadds(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator+=(I other)
        {
            bli_dzadds(other, val);
            return *this;
        }

        Complex& operator-=(const scomplex& other)
        {
            bli_czsubs(other, val);
            return *this;
        }

        Complex& operator-=(const dcomplex& other)
        {
            bli_zsubs(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator-=(const Complex<T>& other)
        {
            bli_zsubs(other.val, val);
            return *this;
        }

        Complex& operator-=(float other)
        {
            bli_szsubs(other, val);
            return *this;
        }

        Complex& operator-=(double other)
        {
            bli_dzsubs(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator-=(I other)
        {
            bli_dzsubs(other, val);
            return *this;
        }

        Complex& operator*=(const scomplex& other)
        {
            bli_czscals(other, val);
            return *this;
        }

        Complex& operator*=(const dcomplex& other)
        {
            bli_zscals(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator*=(const Complex<T>& other)
        {
            bli_zscals(other.val, val);
            return *this;
        }

        Complex& operator*=(float other)
        {
            bli_szscals(other, val);
            return *this;
        }

        Complex& operator*=(double other)
        {
            bli_dzscals(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator*=(I other)
        {
            bli_dzscals(other, val);
            return *this;
        }

        Complex& operator/=(const scomplex& other)
        {
            bli_czinvscals(other, val);
            return *this;
        }

        Complex& operator/=(const dcomplex& other)
        {
            bli_zinvscals(other, val);
            return *this;
        }

        template <typename T>
        Complex& operator/=(const Complex<T>& other)
        {
            bli_zinvscals(other.val, val);
            return *this;
        }

        Complex& operator/=(float other)
        {
            bli_szinvscals(other, val);
            return *this;
        }

        Complex& operator/=(double other)
        {
            bli_dzinvscals(other, val);
            return *this;
        }

        template <class I>
        typename std::enable_if<std::is_integral<I>::value,Complex&>::type
        operator/=(I other)
        {
            bli_dzinvscals(other, val);
            return *this;
        }
};

typedef Complex< float> sComplex;
typedef Complex<double> dComplex;

inline Complex< float> operator+(                float  b, const        scomplex& a) { return Complex< float>(a) += b; }
inline Complex< float> operator+(                float  b, const Complex< float>& a) { return Complex< float>(a) += b; }
inline Complex<double> operator+(                float  b, const        dcomplex& a) { return Complex<double>(a) += b; }
inline Complex<double> operator+(                float  b, const Complex<double>& a) { return Complex<double>(a) += b; }
inline Complex<double> operator+(               double  b, const        scomplex& a) { return Complex<double>(a) += b; }
inline Complex<double> operator+(               double  b, const Complex< float>& a) { return Complex<double>(a) += b; }
inline Complex<double> operator+(               double  b, const        dcomplex& a) { return Complex<double>(a) += b; }
inline Complex<double> operator+(               double  b, const Complex<double>& a) { return Complex<double>(a) += b; }
inline Complex< float> operator+(const        scomplex& a,                 float  b) { return Complex< float>(a) += b; }
inline Complex< float> operator+(const        scomplex& a, const        scomplex& b) { return Complex< float>(a) += b; }
inline Complex< float> operator+(const        scomplex& a, const Complex< float>& b) { return Complex< float>(a) += b; }
inline Complex<double> operator+(const        scomplex& a,                double  b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        scomplex& a, const        dcomplex& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        scomplex& a, const Complex<double>& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        dcomplex& a,                 float  b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        dcomplex& a,                double  b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        dcomplex& a, const        scomplex& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        dcomplex& a, const        dcomplex& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        dcomplex& a, const Complex< float>& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const        dcomplex& a, const Complex<double>& b) { return Complex<double>(a) += b; }
inline Complex< float> operator+(const Complex< float>& a,                 float  b) { return Complex< float>(a) += b; }
inline Complex< float> operator+(const Complex< float>& a, const        scomplex& b) { return Complex< float>(a) += b; }
inline Complex< float> operator+(const Complex< float>& a, const Complex< float>& b) { return Complex< float>(a) += b; }
inline Complex<double> operator+(const Complex< float>& a,                double  b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex< float>& a, const        dcomplex& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex< float>& a, const Complex<double>& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex<double>& a,                 float  b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex<double>& a,                double  b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex<double>& a, const        scomplex& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex<double>& a, const        dcomplex& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex<double>& a, const Complex< float>& b) { return Complex<double>(a) += b; }
inline Complex<double> operator+(const Complex<double>& a, const Complex<double>& b) { return Complex<double>(a) += b; }

inline Complex< float> operator-(                float  a, const        scomplex& b) { return Complex< float>(a) -= b; }
inline Complex< float> operator-(                float  a, const Complex< float>& b) { return Complex< float>(a) -= b; }
inline Complex<double> operator-(                float  a, const        dcomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(                float  a, const Complex<double>& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(               double  a, const        scomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(               double  a, const Complex< float>& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(               double  a, const        dcomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(               double  a, const Complex<double>& b) { return Complex<double>(a) -= b; }
inline Complex< float> operator-(const        scomplex& a,                 float  b) { return Complex< float>(a) -= b; }
inline Complex< float> operator-(const        scomplex& a, const        scomplex& b) { return Complex< float>(a) -= b; }
inline Complex< float> operator-(const        scomplex& a, const Complex< float>& b) { return Complex< float>(a) -= b; }
inline Complex<double> operator-(const        scomplex& a,                double  b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        scomplex& a, const        dcomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        scomplex& a, const Complex<double>& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        dcomplex& a,                 float  b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        dcomplex& a,                double  b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        dcomplex& a, const        scomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        dcomplex& a, const        dcomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        dcomplex& a, const Complex< float>& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const        dcomplex& a, const Complex<double>& b) { return Complex<double>(a) -= b; }
inline Complex< float> operator-(const Complex< float>& a,                 float  b) { return Complex< float>(a) -= b; }
inline Complex< float> operator-(const Complex< float>& a, const        scomplex& b) { return Complex< float>(a) -= b; }
inline Complex< float> operator-(const Complex< float>& a, const Complex< float>& b) { return Complex< float>(a) -= b; }
inline Complex<double> operator-(const Complex< float>& a,                double  b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex< float>& a, const        dcomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex< float>& a, const Complex<double>& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex<double>& a,                 float  b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex<double>& a,                double  b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex<double>& a, const        scomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex<double>& a, const        dcomplex& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex<double>& a, const Complex< float>& b) { return Complex<double>(a) -= b; }
inline Complex<double> operator-(const Complex<double>& a, const Complex<double>& b) { return Complex<double>(a) -= b; }

inline Complex< float> operator*(                float  b, const        scomplex& a) { return Complex< float>(a) *= b; }
inline Complex< float> operator*(                float  b, const Complex< float>& a) { return Complex< float>(a) *= b; }
inline Complex<double> operator*(                float  b, const        dcomplex& a) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(                float  b, const Complex<double>& a) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(               double  b, const        scomplex& a) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(               double  b, const Complex< float>& a) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(               double  b, const        dcomplex& a) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(               double  b, const Complex<double>& a) { return Complex<double>(a) *= b; }
inline Complex< float> operator*(const        scomplex& a,                 float  b) { return Complex< float>(a) *= b; }
inline Complex< float> operator*(const        scomplex& a, const        scomplex& b) { return Complex< float>(a) *= b; }
inline Complex< float> operator*(const        scomplex& a, const Complex< float>& b) { return Complex< float>(a) *= b; }
inline Complex<double> operator*(const        scomplex& a,                double  b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        scomplex& a, const        dcomplex& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        scomplex& a, const Complex<double>& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        dcomplex& a,                 float  b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        dcomplex& a,                double  b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        dcomplex& a, const        scomplex& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        dcomplex& a, const        dcomplex& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        dcomplex& a, const Complex< float>& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const        dcomplex& a, const Complex<double>& b) { return Complex<double>(a) *= b; }
inline Complex< float> operator*(const Complex< float>& a,                 float  b) { return Complex< float>(a) *= b; }
inline Complex< float> operator*(const Complex< float>& a, const        scomplex& b) { return Complex< float>(a) *= b; }
inline Complex< float> operator*(const Complex< float>& a, const Complex< float>& b) { return Complex< float>(a) *= b; }
inline Complex<double> operator*(const Complex< float>& a,                double  b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex< float>& a, const        dcomplex& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex< float>& a, const Complex<double>& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex<double>& a,                 float  b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex<double>& a,                double  b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex<double>& a, const        scomplex& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex<double>& a, const        dcomplex& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex<double>& a, const Complex< float>& b) { return Complex<double>(a) *= b; }
inline Complex<double> operator*(const Complex<double>& a, const Complex<double>& b) { return Complex<double>(a) *= b; }

inline Complex< float> operator/(                float  a, const        scomplex& b) { return Complex< float>(a) /= b; }
inline Complex< float> operator/(                float  a, const Complex< float>& b) { return Complex< float>(a) /= b; }
inline Complex<double> operator/(                float  a, const        dcomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(                float  a, const Complex<double>& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(               double  a, const        scomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(               double  a, const Complex< float>& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(               double  a, const        dcomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(               double  a, const Complex<double>& b) { return Complex<double>(a) /= b; }
inline Complex< float> operator/(const        scomplex& a,                 float  b) { return Complex< float>(a) /= b; }
inline Complex< float> operator/(const        scomplex& a, const        scomplex& b) { return Complex< float>(a) /= b; }
inline Complex< float> operator/(const        scomplex& a, const Complex< float>& b) { return Complex< float>(a) /= b; }
inline Complex<double> operator/(const        scomplex& a,                double  b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        scomplex& a, const        dcomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        scomplex& a, const Complex<double>& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        dcomplex& a,                 float  b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        dcomplex& a,                double  b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        dcomplex& a, const        scomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        dcomplex& a, const        dcomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        dcomplex& a, const Complex< float>& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const        dcomplex& a, const Complex<double>& b) { return Complex<double>(a) /= b; }
inline Complex< float> operator/(const Complex< float>& a,                 float  b) { return Complex< float>(a) /= b; }
inline Complex< float> operator/(const Complex< float>& a, const        scomplex& b) { return Complex< float>(a) /= b; }
inline Complex< float> operator/(const Complex< float>& a, const Complex< float>& b) { return Complex< float>(a) /= b; }
inline Complex<double> operator/(const Complex< float>& a,                double  b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex< float>& a, const        dcomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex< float>& a, const Complex<double>& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex<double>& a,                 float  b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex<double>& a,                double  b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex<double>& a, const        scomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex<double>& a, const        dcomplex& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex<double>& a, const Complex< float>& b) { return Complex<double>(a) /= b; }
inline Complex<double> operator/(const Complex<double>& a, const Complex<double>& b) { return Complex<double>(a) /= b; }

template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator+(const Complex<T>& f, I i) { return f+T(i); }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator+(I i, const Complex<T>& f) { return T(i)+f; }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator-(const Complex<T>& f, I i) { return f-T(i); }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator-(I i, const Complex<T>& f) { return T(i)-f; }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator*(const Complex<T>& f, I i) { return f*T(i); }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator*(I i, const Complex<T>& f) { return T(i)*f; }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator/(const Complex<T>& f, I i) { return f/T(i); }
template <class T, class I> typename std::enable_if<std::is_integral<I>::value,Complex<T>>::type operator/(I i, const Complex<T>& f) { return T(i)/f; }

template <typename T>
std::ostream& operator<<(std::ostream& os, const Complex<T>& x)
{
    os << "(" << x.real() << "," << x.imag() << ")";
    return os;
}

template <typename T> struct real_type             { typedef T type; };
template <typename T> struct real_type<Complex<T>> { typedef T type; };

template <typename T> struct complex_type             { typedef Complex<T> type; };
template <typename T> struct complex_type<Complex<T>> { typedef Complex<T> type; };

template <typename T> struct datatype;
template <> struct datatype<          float> { static const num_t value =    BLIS_FLOAT; };
template <> struct datatype<         double> { static const num_t value =   BLIS_DOUBLE; };
template <> struct datatype<       scomplex> { static const num_t value = BLIS_SCOMPLEX; };
template <> struct datatype<       dcomplex> { static const num_t value = BLIS_DCOMPLEX; };
template <> struct datatype<Complex< float>> { static const num_t value = BLIS_SCOMPLEX; };
template <> struct datatype<Complex<double>> { static const num_t value = BLIS_DCOMPLEX; };

template <typename T> struct is_real             { static const bool value =  true; };
template <typename T> struct is_real<Complex<T>> { static const bool value = false; };

template <typename T> struct is_complex             { static const bool value = false; };
template <typename T> struct is_complex<Complex<T>> { static const bool value =  true; };

template <typename T>
typename std::enable_if<!is_complex<T>::value,T>::type
conj(T val) { return val; }

template <typename T>
typename std::enable_if<is_complex<T>::value,T>::type
conj(const T& val) { return val.conj(); }

template <typename T>
typename std::enable_if<!is_complex<T>::value,T>::type
real_part(T val) { return val; }

template <typename T>
typename std::enable_if<!is_complex<T>::value,T>::type
imag_part(T val) { return T(); }

template <typename T>
typename std::enable_if<is_complex<T>::value,typename real_type<T>::type>::type
real_part(const T& val) { return val.real(); }

template <typename T>
typename std::enable_if<is_complex<T>::value,typename real_type<T>::type>::type
imag_part(const T& val) { return val.imag(); }

template <typename T, typename U>
bool operator==(const Complex<T>& a, U b)
{
    return a.real() == b && a.imag() == U();
}

template <typename T, typename U>
bool operator!=(const Complex<T>& a, U b)
{
    return !(a == b);
}

template <typename T, typename U>
bool operator==(T a, const Complex<U>& b)
{
    return a == b.real() && T() == b.imag();
}

template <typename T, typename U>
bool operator!=(T a, const Complex<U>& b)
{
    return !(a == b);
}

template <typename T, typename U>
bool operator==(const Complex<T>& a, const Complex<U>& b)
{
    return a.real() == b.real() && a.imag() == b.imag();
}

template <typename T, typename U>
bool operator!=(const Complex<T>& a, const Complex<U>& b)
{
    return !(a == b);
}

template <typename T, typename U>
bool operator<(const Complex<T>& a, U b)
{
    return a.real()+a.imag() < b;
}

template <typename T, typename U>
bool operator>(const Complex<T>& a, U b)
{
    return b < a;
}

template <typename T, typename U>
bool operator<=(const Complex<T>& a, U b)
{
    return !(b < a);
}

template <typename T, typename U>
bool operator>=(const Complex<T>& a, U b)
{
    return !(a < b);
}

template <typename T, typename U>
bool operator<(T a, const Complex<U>& b)
{
    return a < b.real()+b.imag();
}

template <typename T, typename U>
bool operator>(T a, const Complex<U>& b)
{
    return b < a;
}

template <typename T, typename U>
bool operator<=(T a, const Complex<U>& b)
{
    return !(b < a);
}

template <typename T, typename U>
bool operator>=(T a, const Complex<U>& b)
{
    return !(a < b);
}

template <typename T, typename U>
bool operator<(const Complex<T>& a, const Complex<U>& b)
{
    return a.real()+a.imag() < b.real()+b.imag();
}

template <typename T, typename U>
bool operator>(const Complex<T>& a, const Complex<U>& b)
{
    return b < a;
}

template <typename T, typename U>
bool operator<=(const Complex<T>& a, const Complex<U>& b)
{
    return !(b < a);
}

template <typename T, typename U>
bool operator>=(const Complex<T>& a, const Complex<U>& b)
{
    return !(a < b);
}

template <typename T>
typename real_type<T>::type norm2(const Complex<T>& x)
{
    return x.real()*x.real() + x.imag()*x.imag();
}

} namespace std {

template <typename T>
typename blis::real_type<T>::type abs(const blis::Complex<T>& x)
{
    return sqrt(blis::norm2(x));
}

} namespace blis {

template <typename T>
Complex<T> sqrt(const Complex<T>& x)
{
    typedef typename real_type<T>::type R;
    R r = x.real();
    R i = x.imag();
    R m = std::abs(x);
    return Complex<T>(::sqrt((r+m)/2), ::sqrt((r-m)/2)*sgn(i));
}

template <typename T>
class Matrix : private obj_t
{
    public:
        typedef T type;
        typedef typename real_type<T>::type real_type;

    private:
        bool is_view;

    protected:
        void create()
        {
            is_view = false;
            memset(static_cast<obj_t*>(this), 0, sizeof(obj_t));
        }

        void create(const Matrix& other)
        {
            is_view = false;

            create(other.getDataType(), other.getNumRows(), other.getNumCols(),
                   other.getRowStride(), other.getColStride());

            bli_copym(other, this);
        }

        void create(real_type r, real_type i)
        {
            is_view = false;
            bli_obj_scalar_init_detached(datatype<T>::value, this);
            bli_setsc((double)r, (double)i, this);
        }

        void create(dim_t m, dim_t n, inc_t rs, inc_t cs)
        {
            is_view = false;
            bli_obj_create(datatype<T>::value, m, n, rs, cs, this);
        }

        void create(dim_t m, dim_t n, type* p, inc_t rs, inc_t cs)
        {
            is_view = true;
            bli_obj_create_with_attached_buffer(datatype<T>::value, m, n, p, rs, cs, this);
        }

        void free()
        {
            if (!is_view) bli_obj_free(this);
        }

    public:
        Matrix(const Matrix& other)
        {
            create(other);
        }

        explicit Matrix(real_type r = real_type(), real_type i = real_type())
        {
            create(r, i);
        }

        template <typename type_>
        explicit Matrix(type_ val, typename std::enable_if<is_complex<type_>::value &&
                                                           std::is_same<type,type_>::value>::type* = 0)
        {
            create(real_part(val), imag_part(val));
        }

        Matrix(dim_t m, dim_t n)
        {
            create(m, n, 1, m);
        }

        Matrix(dim_t m, dim_t n, inc_t rs, inc_t cs)
        {
            create(m, n, rs, cs);
        }

        explicit Matrix(type* p)
        {
            create(1, 1, p, 1, 1);
        }

        Matrix(dim_t m, dim_t n, type* p)
        {
            create(m, n, p, 1, m);
        }

        Matrix(dim_t m, dim_t n, type* p, inc_t rs, inc_t cs)
        {
            create(m, n, p, rs, cs);
        }

        ~Matrix()
        {
            free();
        }

        Matrix& operator=(const Matrix& other)
        {
            reset(other);
            return *this;
        }

        void reset()
        {
            free();
            create();
        }

        void reset(const Matrix& other)
        {
            if (&other == this) return;
            free();
            create(other);
        }

        void reset(real_type r = real_type(), real_type i = real_type())
        {
            free();
            create(r, i);
        }

        void reset(num_t dt, type val)
        {
            free();
            create(real_part(val), imag_part(val));
        }

        void reset(dim_t m, dim_t n)
        {
            free();
            create(m, n, 1, m);
        }

        void reset(dim_t m, dim_t n, inc_t rs, inc_t cs)
        {
            free();
            create(m, n, rs, cs);
        }

        void reset(type* p)
        {
            free();
            create(1, 1, p, 1, 1);
        }

        void reset(dim_t m, dim_t n, type* p)
        {
            free();
            create(m, n, p, 1, m);
        }

        void reset(dim_t m, dim_t n, type* p, inc_t rs, inc_t cs)
        {
            free();
            create(m, n, p, rs, cs);
        }

        dim_t getNumRows() const
        {
            return bli_obj_length(*this);
        }

        dim_t getNumCols() const
        {
            return bli_obj_width(*this);
        }

        inc_t getRowStride() const
        {
            return bli_obj_row_stride(*this);
        }

        inc_t getColStride() const
        {
            return bli_obj_col_stride(*this);
        }

        num_t getDataType() const
        {
            return bli_obj_datatype(*this);
        }

        type* getBuffer()
        {
            return bli_obj_buffer(*this);
        }

        const type* getBuffer() const
        {
            return bli_obj_buffer(*this);
        }

        operator type*()
        {
            return getBuffer();
        }

        operator const type*() const
        {
            return getBuffer();
        }

        void setTrans(trans_t trans)
        {
            bli_obj_set_onlytrans(trans, *this);
        }

        void setConjTrans(trans_t conjtrans)
        {
            bli_obj_set_conjtrans(conjtrans, *this);
        }

        operator obj_t*()
        {
            return this;
        }

        operator const obj_t*() const
        {
            return this;
        }
};

template <typename T>
class RowVector : public Matrix<T>
{
    public:
        using typename Matrix<T>::type;
        using typename Matrix<T>::real_type;

        RowVector(const RowVector& other)
        : Matrix<T>(other) {}

        explicit RowVector(real_type r = real_type(), real_type i = real_type())
        : Matrix<T>(r, i) {}

        template <typename type_>
        explicit RowVector(type_ val, typename std::enable_if<is_complex<type_>::value &&
                                                              std::is_same<type,type_>::value>::type* = 0)
        : Matrix<T>(val) {}

        explicit RowVector(dim_t n)
        : Matrix<T>(n, 1) {}

        RowVector(dim_t n, inc_t inc)
        : Matrix<T>(n, 1, inc, inc*n) {}

        RowVector(dim_t n, type* p)
        : Matrix<T>(n, 1, p, 1, n) {}

        RowVector(dim_t n, type* p, inc_t inc)
        : Matrix<T>(n, 1, p, inc, inc*n) {}

        RowVector& operator=(const RowVector& other)
        {
            Matrix<T>::operator=(other);
            return *this;
        }
};

template <typename T>
class ColVector : public Matrix<T>
{
    public:
        using typename Matrix<T>::type;
        using typename Matrix<T>::real_type;

        ColVector(const ColVector& other)
        : Matrix<T>(other) {}

        explicit ColVector(real_type r = real_type(), real_type i = real_type())
        : Matrix<T>(r, i) {}

        template <typename type_>
        explicit ColVector(type_ val, typename std::enable_if<is_complex<type_>::value &&
                                                              std::is_same<type,type_>::value>::type* = 0)
        : Matrix<T>(val) {}

        explicit ColVector(dim_t n)
        : Matrix<T>(1, n) {}

        ColVector(dim_t n, inc_t inc)
        : Matrix<T>(1, n, 1, inc) {}

        ColVector(dim_t n, type* p)
        : Matrix<T>(1, n, p, 1, 1) {}

        ColVector(dim_t n, type* p, inc_t inc)
        : Matrix<T>(1, n, p, 1, inc) {}

        ColVector& operator=(const ColVector& other)
        {
            Matrix<T>::operator=(other);
            return *this;
        }
};

template <typename T>
class Scalar : public Matrix<T>
{
    public:
        using typename Matrix<T>::type;
        using typename Matrix<T>::real_type;

        Scalar(const Scalar& other)
        : Matrix<T>(other) {}

        explicit Scalar(real_type r = real_type(), real_type i = real_type())
        : Matrix<T>(r, i) {}

        template <typename type_>
        explicit Scalar(type_ val, typename std::enable_if<is_complex<type_>::value &&
                                                           std::is_same<type,type_>::value>::type* = 0)
        : Matrix<T>(val) {}

        explicit Scalar(type* p)
        : Matrix<T>(p) {}

        Scalar& operator=(const Scalar& other)
        {
            Matrix<T>::operator=(other);
            return *this;
        }
};

}

#endif
