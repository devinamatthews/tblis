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

            typedef T value_type;
            typedef T size_type;

        public:
            class iterator : std::iterator<std::random_access_iterator_tag,T>
            {
                protected:
                    T val;

                public:
                    using typename std::iterator<std::random_access_iterator_tag,T>::iterator_category;
                    using typename std::iterator<std::random_access_iterator_tag,T>::value_type;
                    using typename std::iterator<std::random_access_iterator_tag,T>::difference_type;
                    using typename std::iterator<std::random_access_iterator_tag,T>::pointer;
                    using typename std::iterator<std::random_access_iterator_tag,T>::reference;

                    constexpr iterator() : val(0) {}

                    constexpr iterator(T val) : val(val) {}

                    bool operator==(const iterator& other)
                    {
                        return val == other.val;
                    }

                    bool operator!=(const iterator& other)
                    {
                        return val != other.val;
                    }

                    value_type operator*() const
                    {
                        return val;
                    }

                    iterator& operator++()
                    {
                        ++val;
                        return *this;
                    }

                    iterator operator++(int x)
                    {
                        iterator old(*this);
                        ++val;
                        return old;
                    }

                    iterator& operator--()
                    {
                        --val;
                        return *this;
                    }

                    iterator operator--(int x)
                    {
                        iterator old(*this);
                        --val;
                        return old;
                    }

                    iterator& operator+=(difference_type n)
                    {
                        val += n;
                        return *this;
                    }

                    iterator operator+(difference_type n)
                    {
                        return iterator(val+n);
                    }

                    friend iterator operator+(difference_type n, const iterator& i)
                    {
                        return iterator(i.val+n);
                    }

                    iterator& operator-=(difference_type n)
                    {
                        val -= n;
                        return *this;
                    }

                    iterator operator-(difference_type n)
                    {
                        return iterator(val-n);
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
                        return val+n;
                    }

                    friend void swap(iterator& a, iterator& b)
                    {
                        using std::swap;
                        swap(a.val, b.val);
                    }
            };

            constexpr range_t(T from, T to) : from(from), to(to) {}

            size_type size() const
            {
                return to-from;
            }

            iterator begin() const
            {
                return iterator(from);
            }

            iterator end() const
            {
                return iterator(to);
            }

            value_type front() const
            {
                return from;
            }

            value_type back() const
            {
                return to-1;
            }

            value_type operator[](size_type n) const
            {
                return from+n;
            }

            operator std::vector<T>() const
            {
                return std::vector<T>(begin(), end());
            }
    };

    template <typename T>
    range_t<T> range(T to)
    {
        return range_t<T>(T(), to);
    }

    template <typename T>
    range_t<T> range(T from, T to)
    {
        return range_t<T>(from, to);
    }
}

#endif
