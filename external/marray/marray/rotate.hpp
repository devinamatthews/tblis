#ifndef MARRAY_ROTATE_HPP
#define MARRAY_ROTATE_HPP

#include "marray.hpp"

namespace MArray
{

template <typename T, int N>
void rotate(const marray_view<T, N>& array, const array_1d<len_type>& shift)
{
    MARRAY_ASSERT(shift.size() == array.dimension());

    len_vector shift_;
    shift.slurp(shift_);

    for (auto i : range(array.dimension()))
        rotate(array, i, shift_[i]);
}

template <typename T, int N, typename A>
void rotate(marray<T, N, A>& array, const array_1d<len_type>& shift)
{
    rotate(array.view(), shift);
}

template <typename T, int N, int I, typename... D>
void rotate(const marray_slice<T, N, I, D...>& array, const array_1d<len_type>& shift)
{
    rotate(array.view(), shift);
}

template <typename T, int N>
void rotate(const marray_view<T, N>& array, int dim, len_type shift)
{
    MARRAY_ASSERT(dim >= 0 && dim < array.dimension());

    len_type n = array.length(dim);
    stride_type s = array.stride(dim);

    if (n == 0) return;

    shift = shift%n;
    if (shift < 0) shift += n;

    if (shift == 0) return;

    auto len = array.lengths();
    auto& stride = array.strides();
    len[dim] = 1;

    auto p = array.data();
    auto it = make_iterator(len, stride);
    while (it.next(p))
    {
        auto a = p;
        auto b = p+(shift-1)*s;
        while (a < b)
        {
            std::iter_swap(a, b);
            a += s;
            b -= s;
        }

        a = p+shift*s;
        b = p+(n-1)*s;
        while (a < b)
        {
            std::iter_swap(a, b);
            a += s;
            b -= s;
        }

        a = p;
        b = p+(n-1)*s;
        while (a < b)
        {
            std::iter_swap(a, b);
            a += s;
            b -= s;
        }
    }
}

template <typename T, int N, typename A>
void rotate(marray<T, N, A>& array, int dim, len_type shift)
{
    rotate(array.view(), dim, shift);
}

template <typename T, int N, int I, typename... D>
void rotate(const marray_slice<T, N, I, D...>& array, int dim, len_type shift)
{
    rotate(array.view(), dim, shift);
}

}

#endif //MARRAY_ROTATE_HPP
