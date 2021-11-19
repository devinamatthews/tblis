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
        Pred1 p1_;
        Pred2 p2_;

    public:
        binary_or(Pred1 p1, Pred2 p2) : p1_(p1), p2_(p2) {}

        template <typename T>
        bool operator()(const T& t) const
        {
            return p1_(t) || p2_(t);
        }
};

template<class Pred1, class Pred2>
class binary_and
{
    protected:
        Pred1 p1_;
        Pred2 p2_;

    public:
        binary_and(Pred1 p1, Pred2 p2) : p1_(p1), p2_(p2) {}

        template <typename T>
        bool operator()(const T& t) const
        {
            return p1_(t) && p2_(t);
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

template <class InputIt, class OutputIt, class T>
OutputIt prefix_sum(InputIt first, InputIt last, OutputIt d_first, T init)
{
    if (first == last) return d_first;

    typename std::iterator_traits<InputIt>::value_type sum = init;
    *d_first = sum;

    while (++first != last)
    {
       sum = sum + *first;
       *++d_first = sum;
    }

    return ++d_first;
}

template <class InputIt, class OutputIt, class T, class BinaryOperation>
OutputIt prefix_sum(InputIt first, InputIt last, OutputIt d_first, T init,
                    BinaryOperation op)
{
    if (first == last) return d_first;

    typename std::iterator_traits<InputIt>::value_type sum = init;
    *d_first = sum;

    while (++first != last)
    {
       sum = op(sum, *first);
       *++d_first = sum;
    }

    return ++d_first;
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

template <typename T, typename Predicate>
bool matches(const T& v, Predicate&& pred)
{
    return std::find_if(v.begin(), v.end(), std::forward<Predicate>(pred)) != v.end();
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

template <typename T, typename I>
T& rotate(T& v, I n)
{
    if (n > 0)
    {
        std::rotate(v.begin(), std::next(v.begin(), n), v.end());
    }
    else if (n < 0)
    {
        std::rotate(v.begin(), std::next(v.end(), n), v.end());
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
            *i3 = std::move(*i1);
            ++i1;
            ++i2;
            ++i3;
        }
    }
    v1.erase(i3, v1.end());

    return v1;
}

template <typename T, typename U, typename... Ts>
enable_if_t<(sizeof...(Ts) > 0),T&>
intersect(T& v1, U&& v2, Ts&&... vs)
{
    intersect(v1, std::forward<U>(v2));
    intersect(v1, std::forward<Ts>(vs)...);
    return v1;
}

template <typename T, typename... Ts>
T intersection(T v1, Ts&&... vs)
{
    intersect(v1, std::forward<Ts>(vs)...);
    return v1;
}

template <typename T>
T& unite(T& v1, T v2)
{
    T v3;

    sort(v1);
    sort(v2);

    std::set_union(std::make_move_iterator(v1.begin()),
                   std::make_move_iterator(v1.end()),
                   std::make_move_iterator(v2.begin()),
                   std::make_move_iterator(v2.end()),
                   std::back_inserter(v3));
    v1.swap(v3);

    return v1;
}

template <typename T, typename U, typename... Ts>
enable_if_t<(sizeof...(Ts) > 0),T&>
unite(T& v1, U&& v2, Ts&&... vs)
{
    unite(v1, std::forward<U>(v2));
    unite(v1, std::forward<Ts>(vs)...);
    return v1;
}

template <typename T, typename... Ts>
T union_of(T v1, Ts&&... vs)
{
    unite(v1, std::forward<Ts>(vs)...);
    return v1;
}

template <typename T>
T& exclude(T& v1, T v2)
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
            *i3 = std::move(*i1);
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
            ++i2;
        }
    }
    i3 = std::move(i1, v1.end(), i3);
    v1.erase(i3, v1.end());

    return v1;
}

template <typename T, typename U, typename... Ts>
enable_if_t<(sizeof...(Ts) > 0),T&>
exclude(T& v1, U&& v2, Ts&&... vs)
{
    exclude(v1, std::forward<U>(v2));
    exclude(v1, std::forward<Ts>(vs)...);
    return v1;
}

template <typename T, typename... Ts>
T exclusion(T v1, Ts&&... vs)
{
    exclude(v1, std::forward<Ts>(vs)...);
    return v1;
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

template <typename T, typename U>
T permuted(const T& v, const U& p)
{
    T v2; v2.reserve(v.size());
    for (auto& i : p) v2.push_back(v[i]);
    return v2;
}

template <typename T, typename U>
void permute(T& v, const U& p)
{
    v = permuted(v, p);
}

template <typename T, typename U>
T unpermuted(const T& v, const U& p)
{
    T v2; v2.reserve(v.size());
    for (size_t i = 0;i < v.size();i++)
    {
        size_t j = std::find(p.begin(), p.end(), i) - p.begin();
        v2.push_back(v[j]);
    }
    return v2;
}

template <typename T, typename U>
void unpermute(T& v, const U& p)
{
    v = unpermuted(v, p);
}

template <typename T, typename U>
T select_from(const T& v, const U& s, const U& match)
{
    T v2; v2.reserve(match.size());

    for (auto& m : match)
    {
        for (size_t i = 0;i < s.size();i++)
        {
            if (s[i] == m)
            {
                v2.push_back(v[i]);
                break;
            }
        }
    }

    return v2;
}

template <typename T, typename U>
T select_from(const T& v, const U& idx)
{
    T v2; v2.reserve(idx.size());
    for (auto& i : idx) v2.push_back(v[i]);
    return v2;
}

template <typename T, typename U>
void append(T& t, const U& u)
{
    t.insert(t.end(), u.begin(), u.end());
}

template <typename T, typename U, typename V, typename... W>
void append(T& t, const U& u, const V& v, const W&... w)
{
    append(t, u);
    append(t, v, w...);
}

template <typename T, typename... U>
T appended(T t, const U&... u)
{
    append(t, u...);
    return t;
}

}

#endif
