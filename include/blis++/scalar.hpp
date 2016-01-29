#ifndef _TENSOR_BLIS___SCALAR_HPP_
#define _TENSOR_BLIS___SCALAR_HPP_

#include "matrix.hpp"

namespace blis
{

template <typename T>
class Scalar : public Matrix<T>
{
    public:
        using typename Matrix<T>::type;
        using typename Matrix<T>::real_type;

        Scalar(const Scalar& other)
        : Matrix<type>(other) {}

        explicit Scalar(real_type r = real_type(), real_type i = real_type())
        : Matrix<type>(r, i) {}

        template <typename type_>
        explicit Scalar(const type_& val, typename detail::if_same_and_complex<type,type_>::type* = 0)
        : Matrix<type>(val) {}

        explicit Scalar(type* p)
        : Matrix<type>(p) {}

        Scalar& operator=(const Scalar& other)
        {
            Matrix<type>::operator=(other);
            return *this;
        }

        explicit operator type&()
        {
            return *(type*)(*this);
        }

        explicit operator const type&() const
        {
            return *(type*)(*this);
        }
};

}

#endif
