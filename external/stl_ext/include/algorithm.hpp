#ifndef _STL_EXT_ALGORITHM_HPP_
#define _STL_EXT_ALGORITHM_HPP_

#include <algorithm>
#include <iterator>

#include "cosort.hpp"
#include "type_traits.hpp"

namespace stl_ext
{

template<class Pred1, class Pred2>
class binary_or
{
    protected:
        Pred1 p1;
        Pred2 p2;

    public:
        binary_or(Pred1 p1, Pred2 p2) : p1(p1), p2(p2) {}

        template <typename T>
        bool operator()(const T& t) const
        {
            return p1(t)||p2(t);
        }
};

template<class Pred1, class Pred2>
class binary_and
{
    protected:
        Pred1 p1;
        Pred2 p2;

    public:
        binary_and(Pred1 p1, Pred2 p2) : p1(p1), p2(p2) {}

        template <typename T>
        bool operator()(const T& t) const
        {
            return p1(t)&&p2(t);
        }
};

template<class Pred1, class Pred2>
binary_or<Pred1,Pred2> or1(Pred1 p1, Pred2 p2)
{
    return binary_or<Pred1,Pred2>(p1,p2);
}

template<class Pred1, class Pred2>
binary_and<Pred1,Pred2> and1(Pred1 p1, Pred2 p2)
{
    return binary_and<Pred1,Pred2>(p1,p2);
}

template <typename T>
typename T::value_type max(const T& t)
{
    typedef typename T::value_type V;

    if (t.empty()) return V();

    typename T::const_iterator i = t.begin();
    V v = *i;
    for (;i != t.end();++i) if (v < *i) v = *i;

    return v;
}

template <typename T>
typename T::value_type min(const T& t)
{
    typedef typename T::value_type V;

    if (t.empty()) return V();

    typename T::const_iterator i = t.begin();
    V v = *i;
    for (;i != t.end();++i) if (*i < v) v = *i;

    return v;
}

template <typename T, typename Functor>
enable_if_not_same_t<typename T::value_type,Functor,T&>
erase(T& v, const Functor& f)
{
    v.erase(std::remove_if(v.begin(), v.end(), f), v.end());
    return v;
}

template <typename T>
T& erase(T& v, const typename T::value_type& e)
{
    v.erase(std::remove(v.begin(), v.end(), e), v.end());
    return v;
}

template <typename T, typename Functor>
enable_if_not_same_t<typename T::value_type,Functor,T>
erased(T v, const Functor& x)
{
    erase(v, x);
    return v;
}

template <typename T>
T erased(T v, const typename T::value_type& e)
{
    erase(v, e);
    return v;
}

template <typename T, class Predicate>
T& filter(T& v, Predicate pred)
{
    auto i1 = v.begin();
    auto i2 = v.begin();

    while (i1 != v.end())
    {
        if (pred(*i1))
        {
            std::iter_swap(i1, i2);
            ++i2;
        }
        ++i1;
    }

    v.resize(i2-v.begin());
    return v;
}

template <typename T, class Predicate>
T filtered(T v, Predicate pred)
{
    filter(v, pred);
    return v;
}

template <template <typename...> class T, typename U, class Functor>
auto apply(const T<U>& v, const Functor& f) -> T<decltype(f(std::declval<U>()))>
{
    T<decltype(f(std::declval<U>()))> v2;
    for (auto& i : v)
    {
        v2.push_back(f(i));
    }
    return v2;
}

template <typename T>
typename T::value_type sum(const T& v)
{
    typedef typename T::value_type U;
    U s = U();
    for (auto& i : v) s += i;
    return s;
}

template <typename T>
typename T::value_type prod(const T& v)
{
    typedef typename T::value_type U;
    U s = U(1);
    for (auto& i : v) s *= i;
    return s;
}

template <typename T>
bool contains(const T& v, const typename T::value_type& e)
{
    return find(v.begin(), v.end(), e) != v.end();
}

template <typename T>
T& sort(T& v)
{
    std::sort(v.begin(), v.end());
    return v;
}

template <typename T, typename Compare>
T& sort(T& v, const Compare& comp)
{
    std::sort(v.begin(), v.end(), comp);
    return v;
}

template <typename T>
T sorted(T v)
{
    sort(v);
    return v;
}

template <typename T, typename Compare>
T sorted(T v, const Compare& comp)
{
    sort(v, comp);
    return v;
}

template <typename T>
T& unique(T& v)
{
    sort(v);
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

template <typename T>
T uniqued(T v)
{
    unique(v);
    return v;
}

template <typename T>
T& rotate(T& v, typename T::iterator::diff_type n)
{
    if (n > 0)
    {
        std::rotate(v.begin(), std::advance(v.begin(),n), v.end());
    }
    else if (n < 0)
    {
        std::rotate(v.begin(), std::advance(v.end(),n), v.end());
    }
    return v;
}

template <typename T>
T& intersect(T& v1, T v2)
{
    sort(v1);
    sort(v2);

    auto i1 = v1.begin();
    auto i2 = v2.begin();
    auto i3 = v1.begin();
    while (i1 != v1.end() && i2 != v2.end())
    {
        if (*i1 < *i2)
        {
            ++i1;
        }
        else if (*i2 < *i1)
        {
            ++i2;
        }
        else
        {
            std::iter_swap(i1, i3);
            ++i1;
            ++i3;
        }
    }
    v1.resize(i3-v1.begin());

    return v1;
}

template <typename T>
T intersection(const T& v1, const T& v2)
{
    T v3(v1);
    intersect(v3, v2);
    return v3;
}

template <typename T>
T intersection(const T& v1, T&& v2)
{
    T v3(v1);
    intersect(v3, std::move(v2));
    return v3;
}

template <typename T>
T intersection(T&& v1, const T& v2)
{
    T v3(std::move(v1));
    intersect(v3, v2);
    return v3;
}

template <typename T>
enable_if_not_reference_t<T,T>
intersection(T&& v1, T&& v2)
{
    T v3(std::move(v1));
    intersect(v3, std::move(v2));
    return v3;
}

template <typename T>
T& exclude(T& v1, T v2)
{
    sort(v1);
    sort(v2);

    auto i1 = v1.begin();
    auto i2 = v2.begin();
    auto i3 = v1.begin();
    while (i1 != v1.end())
    {
        if (i2 == v2.end() || *i1 < *i2)
        {
            std::iter_swap(i1, i3);
            ++i1;
            ++i3;
        }
        else if (*i2 < *i1)
        {
            ++i2;
        }
        else
        {
            ++i1;
        }
    }
    v1.resize(i3-v1.begin());

    return v1;
}

template <typename T>
T exclusion(const T& v1, const T& v2)
{
    T v3(v1);
    exclude(v3, v2);
    return v3;
}

template <typename T>
T exclusion(const T& v1, T&& v2)
{
    T v3(v1);
    exclude(v3, std::move(v2));
    return v3;
}

template <typename T>
T exclusion(T&& v1, const T& v2)
{
    T v3(std::move(v1));
    exclude(v3, v2);
    return v3;
}

template <typename T>
enable_if_not_reference_t<T,T>
exclusion(T&& v1, T&& v2)
{
    T v3(std::move(v1));
    exclude(v3, std::move(v2));
    return v3;
}

template <typename T>
T mutual_exclusion(T v1, T v2)
{
    T v3;
    sort(v1);
    sort(v2);
    std::set_symmetric_difference(std::make_move_iterator(v1.begin()),
                                  std::make_move_iterator(v1.end()),
                                  std::make_move_iterator(v2.begin()),
                                  std::make_move_iterator(v2.end()),
                                  std::back_inserter(v3));
    return v3;
}

template <typename T, typename U>
T& mask(T& v, const U& m)
{
    auto i1 = v.begin();
    auto i2 = m.begin();
    auto i3 = v.begin();
    while (i1 != v.end())
    {
        if (*i2)
        {
            std::iter_swap(i1, i3);
            ++i3;
        }
        ++i1;
        ++i2;
    }

    v.resize(i3-v.begin());
    return v;
}

template <typename T, typename U>
T masked(T v, const U& m)
{
    mask(v, m);
    return v;
}

template <typename T, typename U>
T& translate(T& s, U from, U to)
{
    cosort(from, to);

    for (auto& l : s)
    {
        auto lb = lower_bound(from.begin(), from.end(), l);

        if (lb != from.end() && *lb == l)
        {
            l = to[lb - from.begin()];
        }
    }

    return s;
}

template <typename T, typename U>
T translated(T s, const U& from, const U& to)
{
    translate(s, from, to);
    return s;
}

template <typename T, typename U>
T translated(T s, const U& from, U&& to)
{
    translate(s, from, std::move(to));
    return s;
}

template <typename T, typename U>
T translated(T s, U&& from, const U& to)
{
    translate(s, std::move(from), to);
    return s;
}

template <typename T, typename U>
enable_if_not_reference_t<U,T>
translated(T s, U&& from, U&& to)
{
    translate(s, std::move(from), std::move(to));
    return s;
}

}

#endif
