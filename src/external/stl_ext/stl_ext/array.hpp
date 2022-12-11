#ifndef _STL_EXT_ARRAY_HPP_
#define _STL_EXT_ARRAY_HPP_

#include <array>

#include "type_traits.hpp"

namespace stl_ext
{

namespace detail
{

template <typename T, size_t N, size_t... M>
struct array_helper
{
    typedef std::array<typename array_helper<T,M...>::type,N> type;
};

template <typename T, size_t N>
struct array_helper<T,N>
{
    typedef std::array<T,N> type;
};

}

template <typename T, size_t... N>
using array = typename detail::array_helper<T,N...>::type;

}

#endif
