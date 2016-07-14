#ifndef _MARRAY_UTILITY_HPP_
#define _MARRAY_UTILITY_HPP_

#include <type_traits>
#include <array>
#include <vector>
#include <utility>
#include <iterator>
#include <cassert>
#include <algorithm>

namespace MArray
{
    /*
     * Create a vector from the specified elements, where the type of the vector
     * is taken from the first element.
     */
    template <typename T, typename... Args>
    std::vector<typename std::decay<T>::type>
    make_vector(T&& t, Args&&... args)
    {
        return {{std::forward<T>(t), std::forward<Args>(args)...}};
    }

    /*
     * Create an array from the specified elements, where the type of the array
     * is taken from the first element.
     */
    template <typename T, typename... Args>
    std::array<typename std::decay<T>::type, sizeof...(Args)+1>
    make_array(T&& t, Args&&... args)
    {
        return {{std::forward<T>(t), std::forward<Args>(args)...}};
    }

    template <typename T>
    class range_t
    {
        static_assert(std::is_integral<T>::value, "The type must be integral.");

        protected:
            T from;
            T to;
            T delta;

            typedef T value_type;
            typedef T size_type;

        public:
            class iterator : std::iterator<std::random_access_iterator_tag,T>
            {
                protected:
                    T val;
                    T delta;

                public:
                    using typename std::iterator<std::random_access_iterator_tag,T>::iterator_category;
                    using typename std::iterator<std::random_access_iterator_tag,T>::value_type;
                    using typename std::iterator<std::random_access_iterator_tag,T>::difference_type;
                    using typename std::iterator<std::random_access_iterator_tag,T>::pointer;
                    using typename std::iterator<std::random_access_iterator_tag,T>::reference;

                    constexpr iterator() : val(0), delta(0) {}

                    constexpr iterator(T val, T delta) : val(val), delta(delta) {}

                    bool operator==(const iterator& other)
                    {
                        return val == other.val && delta == other.delta;
                    }

                    bool operator!=(const iterator& other)
                    {
                        return val != other.val || delta != other.delta;
                    }

                    value_type operator*() const
                    {
                        return val;
                    }

                    iterator& operator++()
                    {
                        val += delta;
                        return *this;
                    }

                    iterator operator++(int x)
                    {
                        iterator old(*this);
                        val += delta;
                        return old;
                    }

                    iterator& operator--()
                    {
                        val -= delta;
                        return *this;
                    }

                    iterator operator--(int x)
                    {
                        iterator old(*this);
                        val -= delta;
                        return old;
                    }

                    iterator& operator+=(difference_type n)
                    {
                        val += n*delta;
                        return *this;
                    }

                    iterator operator+(difference_type n)
                    {
                        return iterator(val+n*delta);
                    }

                    friend iterator operator+(difference_type n, const iterator& i)
                    {
                        return iterator(i.val+n*i.delta);
                    }

                    iterator& operator-=(difference_type n)
                    {
                        val -= n*delta;
                        return *this;
                    }

                    iterator operator-(difference_type n)
                    {
                        return iterator(val-n*delta);
                    }

                    difference_type operator-(const iterator& other)
                    {
                        return val-other.val;
                    }

                    bool operator<(const iterator& other)
                    {
                        return val < other.val;
                    }

                    bool operator<=(const iterator& other)
                    {
                        return val <= other.val;
                    }

                    bool operator>(const iterator& other)
                    {
                        return val > other.val;
                    }

                    bool operator>=(const iterator& other)
                    {
                        return val >= other.val;
                    }

                    value_type operator[](difference_type n) const
                    {
                        return val+n*delta;
                    }

                    friend void swap(iterator& a, iterator& b)
                    {
                        using std::swap;
                        swap(a.val, b.val);
                        swap(a.delta, b.delta);
                    }
            };

            constexpr range_t()
            : from(0), to(0), delta(0) {}

            constexpr range_t(T from, T to, T delta)
            : from(from), to(from+((to-from+delta-1)/delta)*delta), delta(delta) {}

            range_t(const range_t&) = default;

            range_t(range_t&&) = default;

            range_t& operator=(const range_t&) = default;

            range_t& operator=(range_t&&) = default;

            size_type size() const
            {
                return (to-from)/delta;
            }

            iterator begin() const
            {
                return iterator(from, delta);
            }

            iterator end() const
            {
                return iterator(to, delta);
            }

            value_type front() const
            {
                return from;
            }

            value_type back() const
            {
                return to-delta;
            }

            value_type operator[](size_type n) const
            {
                return from+n*delta;
            }

            operator std::vector<T>() const
            {
                return std::vector<T>(begin(), end());
            }
    };

    template <typename T>
    range_t<T> range(T to)
    {
        return {T(), to, 1};
    }

    template <typename T>
    range_t<T> range(T from, T to)
    {
        return {from, to, 1};
    }

    template <typename T>
    range_t<T> range(T from, T to, T delta)
    {
        return {from, to, delta};
    }
}

#endif
