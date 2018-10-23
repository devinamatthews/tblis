#ifndef _MARRAY_DPD_RANGE_HPP_
#define _MARRAY_DPD_RANGE_HPP_

#include "marray_view.hpp"

namespace MArray
{

class index
{
    protected:
        unsigned irrep_;
        len_type idx_;

    public:
        index(unsigned irrep, len_type idx)
        : irrep_(irrep), idx_(idx) {}

        unsigned irrep() const { return irrep_; }

        len_type idx() const { return idx_; }
};

class dpd_range_t
{
    protected:
        std::array<len_type, 8> from_ = {};
        std::array<len_type, 8> to_ = {};

    public:
        dpd_range_t(const detail::array_1d<len_type>& from,
                    const detail::array_1d<len_type>& to)
        {
            from.slurp(from_);
            to.slurp(to_);
        }

        len_type size(unsigned i) const
        {
            MARRAY_ASSERT(i < 8);
            return to_[i] - from_[i];
        }

        len_type from(unsigned i) const
        {
            MARRAY_ASSERT(i < 8);
            return from_[i];
        }

        len_type to(unsigned i) const
        {
            MARRAY_ASSERT(i < 8);
            return to_[i];
        }
};

inline dpd_range_t range(const detail::array_1d<len_type>& to)
{
    return {{}, to};
}

inline dpd_range_t range(const detail::array_1d<len_type>& from,
                         const detail::array_1d<len_type>& to)
{
    return {from, to};
}

namespace detail
{

template <typename T, typename=void>
struct is_dpd_index_or_slice_helper : std::false_type {};

template <>
struct is_dpd_index_or_slice_helper<index> : std::true_type {};

template <>
struct is_dpd_index_or_slice_helper<dpd_range_t> : std::true_type {};

template <typename T>
struct is_dpd_index_or_slice : is_dpd_index_or_slice_helper<typename std::decay<T>::type> {};

template <typename... Args>
struct are_dpd_indices_or_slices;

template<>
struct are_dpd_indices_or_slices<> : std::true_type {};

template <typename Arg, typename... Args>
struct are_dpd_indices_or_slices<Arg, Args...> :
    conditional_t<is_dpd_index_or_slice<Arg>::value,
                  are_dpd_indices_or_slices<Args...>,
                  std::false_type> {};

template <typename... Args>
struct sliced_dimension;

template <>
struct sliced_dimension<>
{
    static constexpr unsigned value = 0;
};

template <typename Arg, typename... Args>
struct sliced_dimension<Arg, Args...>
{
    static constexpr unsigned value = !(std::is_convertible<Arg, int>::value ||
                                      std::is_convertible<Arg, index>::value) +
                                      sliced_dimension<Args...>::value;
};

}

}

#endif
