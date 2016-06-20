#ifndef _STL_EXT_FILL_ITERATOR_HPP_
#define _STL_EXT_FILL_ITERATOR_HPP_

#include <iterator>

#include "type_traits.hpp"

namespace stl_ext
{

template <typename T>
class fill_iterator : std::iterator<std::random_access_iterator_tag,T>
{
    private:
        typedef std::iterator<std::random_access_iterator_tag,T> iterator_base_;

    public:
        using typename iterator_base_::value_type;
        using typename iterator_base_::difference_type;
        using typename iterator_base_::pointer;
        using typename iterator_base_::reference;
        using typename iterator_base_::iterator_category;

        fill_iterator()
        : ptr_(0), idx_(0) {}

        fill_iterator(size_t n, reference ref)
        : ptr_(&ref), idx_(0) {}

        fill_iterator(const fill_iterator& other) = default;

        template <typename U=T, typename=enable_if_const_t<U>>
        fill_iterator(const fill_iterator<remove_const_t<T>>& other)
        : ptr_(other.ptr_), idx_(other.idx_) {}

        fill_iterator& operator=(const fill_iterator& other) = default;

        fill_iterator& operator++()
        {
            ++idx_;
            return *this;
        }

        fill_iterator& operator--()
        {
            --idx_;
            return *this;
        }

        fill_iterator operator++(int x)
        {
            fill_iterator old(*this);
            ++idx_;
            return old;
        }

        fill_iterator operator--(int x)
        {
            fill_iterator old(*this);
            --idx_;
            return old;
        }

        reference operator*()
        {
            return *ptr_;
        }

        pointer operator->()
        {
            return ptr_;
        }

        fill_iterator& operator+=(difference_type n)
        {
            idx_ += n;
            return *this;
        }

        fill_iterator& operator-=(difference_type n)
        {
            idx_ -= n;
            return *this;
        }

        fill_iterator operator+(difference_type n)
        {
            fill_iterator ret(*this);
            ret += n;
            return ret;
        }

        friend fill_iterator operator+(difference_type n, const fill_iterator& x)
        {
            return x+n;
        }

        fill_iterator operator-(difference_type n)
        {
            fill_iterator ret(*this);
            ret -= n;
            return ret;
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        difference_type operator-(const fill_iterator<U>& x)
        {
            return (difference_type)idx_ - (difference_type)x.idx_;
        }

        reference operator[](difference_type i)
        {
            return *ptr_;
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        bool operator==(const fill_iterator<U>& other) const
        {
            return ptr_ == other.ptr_ &&
                   idx_ == other.idx_;
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        bool operator!=(const fill_iterator<U>& other) const
        {
            return !(*this == other);
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        bool operator<(const fill_iterator<U>& other) const
        {
            return ptr_ < other.ptr_ ||
                   (ptr_ == other.ptr_ &&
                    idx_ < other.idx_);
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        bool operator>(const fill_iterator<U>& other) const
        {
            return other < *this;
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        bool operator<=(const fill_iterator<U>& other) const
        {
            return !(other < *this);
        }

        template <typename U, typename=enable_if_similar_t<T,U>>
        bool operator>=(const fill_iterator<U>& other) const
        {
            return !(*this < other);
        }

    private:
        pointer ptr_;
        size_t idx_;
};

template <typename T>
using const_fill_iterator = fill_iterator<const T>;

template <typename T>
fill_iterator<T> fill_begin(size_t n, T& value)
{
    return fill_iterator<T>(n, value);
}

template <typename T>
fill_iterator<T> fill_end(size_t n, T& value)
{
    return fill_iterator<T>(n, value)+n;
}

template <typename T>
const_fill_iterator<T> fill_begin(size_t n, const T& value)
{
    return const_fill_iterator<T>(n, value);
}

template <typename T>
const_fill_iterator<T> fill_end(size_t n, const T& value)
{
    return const_fill_iterator<T>(n, value)+n;
}

template <typename T>
const_fill_iterator<T> fill_cbegin(size_t n, T& value)
{
    return const_fill_iterator<T>(n, value);
}

template <typename T>
const_fill_iterator<T> fill_cend(size_t n, T& value)
{
    return const_fill_iterator<T>(n, value)+n;
}

}

#endif
