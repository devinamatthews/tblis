#ifndef _STL_EXT_VECTOR_HPP_
#define _STL_EXT_VECTOR_HPP_

#include <iterator>
#include <vector>

#include "type_traits.hpp"

namespace stl_ext
{

using std::vector;

namespace detail
{

template <size_t N, typename T, typename U, typename... Ts>
struct vec_helper
{
    vec_helper(vector<T>& v, U&& t, Ts&&... ts)
    {
        v.push_back(std::forward<U>(t));
        vec_helper<N-1, T, Ts...>(v, std::forward<Ts>(ts)...);
    }
};

template <typename T, typename U>
struct vec_helper<1, T, U>
{
    vec_helper(vector<T>& v, U&& t)
    {
        v.push_back(std::forward<U>(t));
    }
};

}

template <typename T, typename... Ts>
vector<decay_t<T>> vec(T&& t, Ts&&... ts)
{
    vector<decay_t<T>> v;
    v.reserve(1+sizeof...(Ts));
    detail::vec_helper<1+sizeof...(Ts), decay_t<T>, T, Ts...>(v, std::forward<T>(t), std::forward<Ts>(ts)...);
    return v;
}

template<typename T>
vector<T> slice(const vector<T>& v, ptrdiff_t e1, ptrdiff_t e2)
{
    if (e1 < 0) e1 += v.size();
    if (e2 < 0) e2 += v.size();
    return vector<T>(v.begin()+e1, v.begin()+e2);
}

template<typename T>
vector<T> slice(vector<T>&& v, ptrdiff_t e1, ptrdiff_t e2)
{
    if (e1 < 0) e1 += v.size();
    if (e2 < 0) e2 += v.size();
    return vector<T>(std::make_move_iterator(v.begin()+e1),
                     std::make_move_iterator(v.begin()+e2));
}

template<typename T>
vector<T> slice(const vector<T>& v, ptrdiff_t e1)
{
    if (e1 < 0) e1 += v.size();
    return vector<T>(v.begin()+e1, v.end());
}

template<typename T>
vector<T> slice(vector<T>&& v, ptrdiff_t e1)
{
    if (e1 < 0) e1 += v.size();
    return vector<T>(std::make_move_iterator(v.begin()+e1),
                     std::make_move_iterator(v.end()));
}

}

namespace std
{

template<typename T> vector<T> operator+(const vector<T>& v1, const vector<T>& v2)
{
    vector<T> r(v1);
    r.insert(r.end(), v2.begin(), v2.end());
    return r;
}

template<typename T> vector<T> operator+(vector<T>&& v1, const vector<T>& v2)
{
    vector<T> r(std::move(v1));
    r.insert(r.end(), v2.begin(), v2.end());
    return r;
}

template<typename T> vector<T> operator+(const vector<T>& v1, vector<T>&& v2)
{
    vector<T> r(v1);
    r.insert(r.end(), std::make_move_iterator(v2.begin()),
                      std::make_move_iterator(v2.end()));
    return r;
}

template<typename T> vector<T> operator+(vector<T>&& v1, vector<T>&& v2)
{
    vector<T> r(std::move(v1));
    r.insert(r.end(), std::make_move_iterator(v2.begin()),
                      std::make_move_iterator(v2.end()));
    return r;
}

template<typename T> vector<T> operator+(const vector<T>& v, const T& t)
{
    vector<T> r(v);
    r.push_back(t);
    return r;
}

template<typename T> vector<T> operator+(vector<T>&& v, const T& t)
{
    vector<T> r(std::move(v));
    r.push_back(t);
    return r;
}

template<typename T> vector<T> operator+(const vector<T>& v, T&& t)
{
    vector<T> r(v);
    r.push_back(std::move(t));
    return r;
}

template<typename T> vector<T> operator+(vector<T>&& v, T&& t)
{
    vector<T> r(std::move(v));
    r.push_back(std::move(t));
    return r;
}

template<typename T>
vector<T>& operator+=(vector<T>& v1, const vector<T>& v2)
{
    v1.insert(v1.end(), v2.begin(), v2.end());
    return v1;
}

template<typename T>
vector<T>& operator+=(vector<T>& v1, vector<T>&& v2)
{
    v1.insert(v1.end(), std::make_move_iterator(v2.begin()),
                        std::make_move_iterator(v2.end()));
    return v1;
}

template<typename T> vector<T>& operator+=(vector<T>& v, const T& t)
{
    v.insert(v.end(), t);
    return v;
}

template<typename T> vector<T>& operator+=(vector<T>& v, T&& t)
{
    v.insert(v.end(), std::move(t));
    return v;
}

template<typename T> bool operator!(const std::vector<T>& x)
{
    return x.empty();
}

}

#endif
