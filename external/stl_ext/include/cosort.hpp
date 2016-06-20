#ifndef _STL_EXT_COSORT_HPP_
#define _STL_EXT_COSORT_HPP_

#include <algorithm>
#include <iterator>

namespace stl_ext
{

namespace detail
{

template <class T, class U>
struct doublet
{
    T first;
    U second;

    doublet(const T& first, const U& second) : first(first), second(second) {}

    friend void swap(doublet& first, doublet& second)
    {
        using std::swap;
        swap(first.first, second.first);
        swap(first.second, second.second);
    }

    friend void swap(doublet&& first, doublet&& second)
    {
        using std::swap;
        swap(first.first, second.first);
        swap(first.second, second.second);
    }

    doublet(const doublet<T&,U&>& other)
    : first(other.first), second(other.second) {}

    doublet(doublet<T&,U&>&& other)
    : first(std::move(other.first)), second(std::move(other.second)) {}

    doublet(const doublet<T,U>& other)
    : first(other.first), second(other.second) {}

    doublet(doublet<T,U>&& other)
    : first(std::move(other.first)), second(std::move(other.second)) {}

    doublet& operator=(const doublet<T&,U&>& other)
    {
        first = other.first;
        second = other.second;
        return *this;
    }

    doublet& operator=(doublet<T&,U&>&& other)
    {
        first = std::move(other.first);
        second = std::move(other.second);
        return *this;
    }

    doublet& operator=(const doublet<T,U>& other)
    {
        first = other.first;
        second = other.second;
        return *this;
    }

    doublet& operator=(doublet<T,U>&& other)
    {
        first = std::move(other.first);
        second = std::move(other.second);
        return *this;
    }

    bool operator==(const doublet<T&,U&>& other) const
    {
        return first == other.first;
    }

    bool operator!=(const doublet<T&,U&>& other) const
    {
        return first != other.first;
    }

    bool operator==(const doublet& other) const
    {
        return first == other.first;
    }

    bool operator!=(const doublet& other) const
    {
        return first != other.first;
    }

    bool operator<(const doublet<T&,U&>& other) const
    {
        return first < other.first;
    }

    bool operator>(const doublet<T&,U&>& other) const
    {
        return first > other.first;
    }

    bool operator<=(const doublet<T&,U&>& other) const
    {
        return first <= other.first;
    }

    bool operator>=(const doublet<T&,U&>& other) const
    {
        return first >= other.first;
    }

    bool operator<(const doublet<T,U>& other) const
    {
        return first < other.first;
    }

    bool operator>(const doublet<T,U>& other) const
    {
        return first > other.first;
    }

    bool operator<=(const doublet<T,U>& other) const
    {
        return first <= other.first;
    }

    bool operator>=(const doublet<T,U>& other) const
    {
        return first >= other.first;
    }
};

template <class T, class U>
struct doublet<T&,U&>
{
    T& first;
    U& second;

    doublet(T& first, U& second) : first(first), second(second) {}

    friend void swap(doublet& first, doublet& second)
    {
        using std::swap;
        swap(first.first, second.first);
        swap(first.second, second.second);
    }

    friend void swap(doublet&& first, doublet&& second)
    {
        using std::swap;
        swap(first.first, second.first);
        swap(first.second, second.second);
    }

    doublet(doublet<T&,U&>& other)
    : first(other.first), second(other.second) {}

    doublet(doublet<T&,U&>&& other)
    : first(other.first), second(other.second) {}

    doublet(doublet<T,U>& other)
    : first(other.first), second(other.second) {}

    doublet& operator=(const doublet<T&,U&>& other)
    {
        first = other.first;
        second = other.second;
        return *this;
    }

    doublet& operator=(doublet<T&,U&>&& other)
    {
        first = std::move(other.first);
        second = std::move(other.second);
        return *this;
    }

    doublet& operator=(const doublet<T,U>& other)
    {
        first = other.first;
        second = other.second;
        return *this;
    }

    doublet& operator=(doublet<T,U>&& other)
    {
        first = std::move(other.first);
        second = std::move(other.second);
        return *this;
    }

    bool operator==(const doublet<T,U>& other) const
    {
        return first == other.first;
    }

    bool operator!=(const doublet<T,U>& other) const
    {
        return first != other.first;
    }

    bool operator==(const doublet& other) const
    {
        return first == other.first;
    }

    bool operator!=(const doublet& other) const
    {
        return first != other.first;
    }

    bool operator<(const doublet<T,U>& other) const
    {
        return first < other.first;
    }

    bool operator>(const doublet<T,U>& other) const
    {
        return first > other.first;
    }

    bool operator<=(const doublet<T,U>& other) const
    {
        return first <= other.first;
    }

    bool operator>=(const doublet<T,U>& other) const
    {
        return first >= other.first;
    }

    bool operator<(const doublet& other) const
    {
        return first < other.first;
    }

    bool operator>(const doublet& other) const
    {
        return first > other.first;
    }

    bool operator<=(const doublet& other) const
    {
        return first <= other.first;
    }

    bool operator>=(const doublet& other) const
    {
        return first >= other.first;
    }
};

template <class T, class U>
class coiterator : public std::iterator<std::random_access_iterator_tag,
//tuple<typename iterator_traits<T>::value_type,
//      typename iterator_traits<U>::value_type>,
//ptrdiff_t,
//tuple<typename iterator_traits<T>::pointer,
//      typename iterator_traits<U>::pointer>,
//tuple<typename iterator_traits<T>::reference,
//      typename iterator_traits<U>::reference>>
doublet<typename std::iterator_traits<T>::value_type,
        typename std::iterator_traits<U>::value_type>,
ptrdiff_t,
doublet<typename std::iterator_traits<T>::pointer,
        typename std::iterator_traits<U>::pointer>,
doublet<typename std::iterator_traits<T>::reference,
        typename std::iterator_traits<U>::reference>>
{
    T it_T;
    U it_U;

    public:
        coiterator(const T& it_T, const U& it_U) : it_T(it_T), it_U(it_U) {}

        bool operator==(const coiterator& other) const
        {
            return it_T == other.it_T;
        }

        bool operator!=(const coiterator& other) const
        {
            return it_T != other.it_T;
        }

        bool operator<(const coiterator& other) const
        {
            return it_T < other.it_T;
        }

        bool operator>(const coiterator& other) const
        {
            return it_T > other.it_T;
        }

        bool operator<=(const coiterator& other) const
        {
            return it_T <= other.it_T;
        }

        bool operator>=(const coiterator& other) const
        {
            return it_T >= other.it_T;
        }

        typename coiterator::reference operator*()
        {
            //return typename coiterator::reference(move(*it_T),move(*it_U));
            return typename coiterator::reference(*it_T,*it_U);
        }

        typename coiterator::reference operator[](ptrdiff_t n)
        {
            //return typename coiterator::reference(move(it_T[n]),move(it_U[n]));
            return typename coiterator::reference(it_T[n],it_U[n]);
        }

        coiterator& operator++()
        {
            ++it_T;
            ++it_U;
            return *this;
        }

        coiterator& operator--()
        {
            --it_T;
            --it_U;
            return *this;
        }

        coiterator operator++(int x)
        {
            return coiterator(it_T++, it_U++);
        }

        coiterator operator--(int x)
        {
            return coiterator(it_T--, it_U--);
        }

        coiterator& operator+=(ptrdiff_t n)
        {
            it_T += n;
            it_U += n;
            return *this;
        }

        coiterator& operator-=(ptrdiff_t n)
        {
            it_T -= n;
            it_U -= n;
            return *this;
        }

        coiterator operator+(ptrdiff_t n) const
        {
            return coiterator(it_T+n, it_U+n);
        }

        friend coiterator operator+(ptrdiff_t n, const coiterator& other)
        {
            return coiterator(other.it_T+n, other.it_U+n);
        }

        coiterator operator-(ptrdiff_t n) const
        {
            return coiterator(it_T-n, it_U-n);
        }

        ptrdiff_t operator-(const coiterator& other) const
        {
            return it_T-other.it_T;
        }
};

template <class key_iterator, class val_iterator, class Comparator>
class cocomparator
{
    typedef typename coiterator<key_iterator,val_iterator>::value_type val;

    Comparator comp;

    public:
        cocomparator(Comparator comp) : comp(comp) {}

        bool operator()(const val& r1, const val& r2) const
        {
            return comp(r1.first, r2.first);
        }
};

}

template <class key_iterator, class val_iterator>
void cosort(key_iterator keys_begin, key_iterator keys_end,
            val_iterator vals_begin, val_iterator vals_end)
{
    detail::coiterator<key_iterator,val_iterator> begin(keys_begin, vals_begin);
    detail::coiterator<key_iterator,val_iterator> end  (keys_end  , vals_end  );
    std::sort(begin, end);
}

template <class key_iterator, class val_iterator, class Comparator>
void cosort(key_iterator keys_begin, key_iterator keys_end,
            val_iterator vals_begin, val_iterator vals_end,
            Comparator comp)
{
    detail::coiterator<key_iterator,val_iterator> begin(keys_begin, vals_begin);
    detail::coiterator<key_iterator,val_iterator> end  (keys_end  , vals_end  );
    std::sort(begin, end, detail::cocomparator<key_iterator,val_iterator,Comparator>(comp));
}

template <class Keys, class Values>
void cosort(Keys& keys, Values& values)
{
    cosort(keys.begin(), keys.end(), values.begin(), values.end());
}

template <class Keys, class Values, class Comparator>
void cosort(Keys& keys, Values& values, Comparator comp)
{
    cosort(keys.begin(), keys.end(), values.begin(), values.end(), comp);
}

}

#endif
