#ifndef _STL_EXT_ZIP_HPP_
#define _STL_EXT_ZIP_HPP_

#include <tuple>
#include <vector>

#include "type_traits.hpp"

namespace stl_ext
{

namespace detail
{

template <size_t I, typename... Args>
struct min_size_helper
{
    size_t operator()(const std::tuple<Args...>& v)
    {
        return std::min(std::get<I-1>(v).size(), min_size_helper<I-1, Args...>()(v));
    }
};

template <typename... Args>
struct min_size_helper<1, Args...>
{
    size_t operator()(const std::tuple<Args...>& v)
    {
        return std::get<0>(v).size();
    }
};

template <typename... Args>
struct min_size_helper<0, Args...>
{
    size_t operator()(const std::tuple<Args...>& v)
    {
        return 0;
    }
};

template <typename... Args>
size_t min_size(const std::tuple<Args...>& v)
{
    return min_size_helper<sizeof...(Args), Args...>()(v);
}

template <size_t I, typename... Args>
struct cbegin_helper
{
    void operator()(std::tuple<typename decay_t<Args>::const_iterator...>& i,
                    const std::tuple<Args...>& v)
    {
        std::get<I-1>(i) = std::get<I-1>(v).begin();
        cbegin_helper<I-1, Args...>()(i, v);
    }
};

template <typename... Args>
struct cbegin_helper<1, Args...>
{
    void operator()(std::tuple<typename decay_t<Args>::const_iterator...>& i,
                    const std::tuple<Args...>& v)
    {
        std::get<0>(i) = std::get<0>(v).begin();
    }
};

template <typename... Args>
struct cbegin_helper<0, Args...>
{
    void operator()(std::tuple<typename decay_t<Args>::const_iterator...>& i,
                    const std::tuple<Args...>& v) {}
};

template <typename... Args>
std::tuple<typename decay_t<Args>::const_iterator...> cbegin(const std::tuple<Args...>& v)
{
    std::tuple<typename decay_t<Args>::const_iterator...> i;
    cbegin_helper<sizeof...(Args), Args...>()(i, v);
    return i;
}

template <size_t I, typename... Args>
struct increment_helper
{
    void operator()(std::tuple<Args...>& i)
    {
        ++std::get<I-1>(i);
        increment_helper<I-1, Args...>()(i);
    }
};

template <typename... Args>
struct increment_helper<1, Args...>
{
    void operator()(std::tuple<Args...>& i)
    {
        ++std::get<0>(i);
    }
};

template <typename... Args>
struct increment_helper<0, Args...>
{
    void operator()(std::tuple<Args...>& i) {}
};

template <typename... Args>
void increment(std::tuple<Args...>& i)
{
    increment_helper<sizeof...(Args), Args...>()(i);
}

template <size_t I, typename... Args>
struct not_end_helper
{
    bool operator()(const std::tuple<typename decay_t<Args>::const_iterator...>& i,
                    const std::tuple<Args...>& v)
    {
        return std::get<I-1>(i) != std::get<I-1>(v).end() &&
               not_end_helper<I-1, Args...>()(i, v);
    }
};

template <typename... Args>
struct not_end_helper<1, Args...>
{
    bool operator()(const std::tuple<typename decay_t<Args>::const_iterator...>& i,
                    const std::tuple<Args...>& v)
    {
        return std::get<0>(i) != std::get<0>(v).end();
    }
};

template <typename... Args>
struct not_end_helper<0, Args...>
{
    bool operator()(const std::tuple<typename decay_t<Args>::const_iterator...>& i,
                    const std::tuple<Args...>& v)
    {
        return false;
    }
};

template <typename... Args>
bool not_end(const std::tuple<typename decay_t<Args>::const_iterator...>& i,
             const std::tuple<Args...>& v)
{
    return not_end_helper<sizeof...(Args), Args...>()(i, v);
}

template <size_t I, typename... Args>
struct reserve_helper
{
    void operator()(std::tuple<Args...>& t, size_t n)
    {
        std::get<I-1>(t).reserve(n);
        reserve_helper<I-1, Args...>()(t, n);
    }
};

template <typename... Args>
struct reserve_helper<1, Args...>
{
    void operator()(std::tuple<Args...>& t, size_t n)
    {
        std::get<0>(t).reserve(n);
    }
};

template <typename... Args>
struct reserve_helper<0, Args...>
{
    void operator()(std::tuple<Args...>& t, size_t n) {}
};

template <typename... Args>
void reserve(std::tuple<Args...>& t, size_t n)
{
    reserve_helper<sizeof...(Args), Args...>()(t, n);
}

template <size_t I, typename... Args>
struct emplace_back_helper
{
    void operator()(std::tuple<Args...>& t, const std::tuple<typename Args::value_type...>& v)
    {
        std::get<I-1>(t).emplace_back(std::get<I-1>(v));
        emplace_back_helper<I-1, Args...>()(t, v);
    }

    void operator()(std::tuple<Args...>& t, std::tuple<typename Args::value_type...>&& v)
    {
        std::get<I-1>(t).emplace_back(std::move(std::get<I-1>(v)));
        emplace_back_helper<I-1, Args...>()(t, v);
    }
};

template <typename... Args>
struct emplace_back_helper<1, Args...>
{
    void operator()(std::tuple<Args...>& t, const std::tuple<typename Args::value_type...>& v)
    {
        std::get<0>(t).emplace_back(std::get<0>(v));
    }

    void operator()(std::tuple<Args...>& t, std::tuple<typename Args::value_type...>&& v)
    {
        std::get<0>(t).emplace_back(std::move(std::get<0>(v)));
    }
};

template <typename... Args>
struct emplace_back_helper<0, Args...>
{
    void operator()(std::tuple<Args...>& t, const std::tuple<typename Args::value_type...>& v) {}

    void operator()(std::tuple<Args...>& t, std::tuple<typename Args::value_type...>&& v) {}
};

template <typename... Args>
void emplace_back(std::tuple<Args...>& t, const std::tuple<typename Args::value_type...>& v)
{
    emplace_back_helper<sizeof...(Args), Args...>()(t, v);
}

template <typename... Args>
void emplace_back(std::tuple<Args...>& t, std::tuple<typename Args::value_type...>&& v)
{
    emplace_back_helper<sizeof...(Args), Args...>()(t, std::move(v));
}

template <typename T, T... S> struct integer_sequence {};

template <typename T, typename U, typename V> struct concat_sequences;
template <typename T, T... S, T... R>
struct concat_sequences<T, integer_sequence<T, S...>, integer_sequence<T, R...>>
{
    typedef integer_sequence<T, S..., (R+sizeof...(S))...> type;
};

template <size_t N> struct static_range_helper;

template <> struct static_range_helper<0>
{
    typedef integer_sequence<size_t> type;
};

template <> struct static_range_helper<1>
{
    typedef integer_sequence<size_t,0> type;
};

template <size_t N> struct static_range_helper
{
    typedef typename concat_sequences<size_t, typename static_range_helper<(N+1)/2>::type,
                                              typename static_range_helper<N/2>::type>::type type;
};

template <size_t N>
using static_range = typename static_range_helper<N>::type;

template <typename... Args>
struct call_helper
{
    template <typename Func, size_t... S>
    call_helper(Func func, std::tuple<Args...>& args, integer_sequence<size_t, S...> seq)
    {
        func(std::get<S>(args)...);
    }

    template <typename Func, size_t... S>
    call_helper(Func func, const std::tuple<Args...>& args, integer_sequence<size_t, S...> seq)
    {
        func(std::get<S>(args)...);
    }

    template <typename Func, size_t... S>
    call_helper(Func func, std::tuple<Args...>&& args, integer_sequence<size_t, S...> seq)
    {
        func(std::get<S>(std::move(args))...);
    }
};

}

template <typename Func, typename... Args>
void call(Func func, std::tuple<Args...>& args)
{
    detail::call_helper<Args...>(func, args, detail::static_range<sizeof...(Args)>{});
}

template <typename Func, typename... Args>
void call(Func func, const std::tuple<Args...>& args)
{
    detail::call_helper<Args...>(func, args, detail::static_range<sizeof...(Args)>{});
}

template <typename Func, typename... Args>
void call(Func func, std::tuple<Args...>&& args)
{
    detail::call_helper<Args...>(func, std::move(args), detail::static_range<sizeof...(Args)>{});
}

template <typename... Args>
std::vector<std::tuple<typename decay_t<Args>::value_type...>> zip(const std::tuple<Args...>& v)
{
    //TODO move elements when possible

    std::vector<std::tuple<typename decay_t<Args>::value_type...>> t;
    t.reserve(detail::min_size(v));

    auto i = detail::cbegin(v);
    for (;detail::not_end(i,v);detail::increment(i))
    {
        call([&t](typename decay_t<Args>::const_iterator... args) {t.emplace_back(*args...); }, i);
    }

    return t;
}

template <typename Arg, typename... Args>
std::vector<std::tuple<typename decay_t<Arg>::value_type, typename decay_t<Args>::value_type...>>
zip(Arg&& v, Args&&... v_)
{
    return zip(forward_as_tuple(std::forward<Arg>(v), std::forward<Args>(v_)...));
}

template <typename... Args>
std::tuple<std::vector<Args>...> unzip(const std::vector<std::tuple<Args...>>& v)
{
    std::tuple<std::vector<Args>...> t;
    detail::reserve(t, v.size());

    for (auto i = v.begin();i != v.end();++i)
    {
        detail::emplace_back(t, *i);
    }

    return t;
}

template <typename... Args>
std::tuple<std::vector<Args>...> unzip(std::vector<std::tuple<Args...>>&& v)
{
    std::tuple<std::vector<Args>...> t;
    detail::reserve(t, v.size());

    for (auto i = v.begin();i != v.end();++i)
    {
        detail::emplace_back(t, move(*i));
    }

    return t;
}

}

#endif
