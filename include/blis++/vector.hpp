#ifndef _TENSOR_BLIS___VECTOR_HPP_
#define _TENSOR_BLIS___VECTOR_HPP_

#include "matrix.hpp"

namespace blis
{

template <typename T>
class RowVector : public Matrix<T>
{
    public:
        using typename Matrix<T>::type;
        using typename Matrix<T>::real_type;

        RowVector() {}

        RowVector(const RowVector& other)
        : Matrix<type>(other) {}

        RowVector(RowVector&& other)
        : Matrix<type>(std::move(other)) {}

        explicit RowVector(dim_t n)
        : Matrix<type>(n, 1) {}

        RowVector(dim_t n, inc_t inc)
        : Matrix<type>(n, 1, inc, inc*n) {}

        RowVector(dim_t n, type* p)
        : Matrix<type>(n, 1, p, 1, n) {}

        RowVector(dim_t n, type* p, inc_t inc)
        : Matrix<type>(n, 1, p, inc, inc*n) {}

        RowVector& operator=(const RowVector& other)
        {
            Matrix<type>::operator=(other);
            return *this;
        }

        RowVector& operator=(RowVector&& other)
        {
            Matrix<type>::operator=(std::move(other));
            return *this;
        }
};

template <typename T>
class ColVector : public Matrix<T>
{
    public:
        using typename Matrix<T>::type;
        using typename Matrix<T>::real_type;

        ColVector() {}

        ColVector(const ColVector& other)
        : Matrix<type>(other) {}

        ColVector(ColVector&& other)
        : Matrix<type>(std::move(other)) {}

        explicit ColVector(dim_t n)
        : Matrix<type>(1, n) {}

        ColVector(dim_t n, inc_t inc)
        : Matrix<type>(1, n, 1, inc) {}

        ColVector(dim_t n, type* p)
        : Matrix<type>(1, n, p, 1, 1) {}

        ColVector(dim_t n, type* p, inc_t inc)
        : Matrix<type>(1, n, p, 1, inc) {}

        ColVector& operator=(const ColVector& other)
        {
            Matrix<type>::operator=(other);
            return *this;
        }

        ColVector& operator=(ColVector&& other)
        {
            Matrix<type>::operator=(std::move(other));
            return *this;
        }
};

}

#endif
