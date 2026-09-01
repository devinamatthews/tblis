#ifndef MARRAY_DPD_RANGE_HPP
#define MARRAY_DPD_RANGE_HPP

#include "../marray_base.hpp"
#include "../range.hpp"

MARRAY_BEGIN_NAMESPACE

class dpd_index
{
  protected:
    int irrep_;
    len_type idx_;

  public:
    dpd_index(int irrep, len_type idx) : irrep_(irrep), idx_(idx) {}

    int irrep() const { return irrep_; }

    len_type idx() const { return idx_; }

    bool operator==(const dpd_index& other) const
    {
        return irrep_ == other.irrep_ && idx_ == other.idx_;
    }

    auto operator<=>(const dpd_index& other) const
    {
        if (irrep_ != other.irrep_)
            return irrep_ <=> other.irrep_;
        return idx_ <=> other.idx_;
    }
};

class dpd_range : public std::array<range_t<len_type>, 8>
{
  public:
    dpd_range() : std::array<range_t<len_type>, 8>{} {}

    dpd_range(const array_1d<len_type>& to_)
    {
        MARRAY_ASSERT(to_.size() <= 8);

        len_vector to;
        to_.slurp(to);

        for (auto i : range(to.size())) (*this)[i] = range(to[i]);
    }

    dpd_range(const array_1d<len_type>& from_, const array_1d<len_type>& to_)
    {
        MARRAY_ASSERT(from_.size() == to_.size());
        MARRAY_ASSERT(from_.size() <= 8);

        len_vector from;
        from_.slurp(from);
        len_vector to;
        to_.slurp(to);

        for (auto i : range(from.size())) (*this)[i] = range(from[i], to[i]);
    }

    dpd_range(const array_1d<len_type>& from_,
              const array_1d<len_type>& to_,
              const array_1d<len_type>& delta_)
    {
        MARRAY_ASSERT(from_.size() == to_.size());
        MARRAY_ASSERT(from_.size() == delta_.size());
        MARRAY_ASSERT(from_.size() <= 8);

        len_vector from;
        from_.slurp(from);
        len_vector to;
        to_.slurp(to);
        len_vector delta;
        delta_.slurp(delta);

        for (auto i : range(from.size()))
            (*this)[i] = range(from[i], to[i], delta[i]);
    }

    dpd_range(int irrep, const range_t<len_type>& x)
    : std::array<range_t<len_type>, 8>{}
    {
        MARRAY_ASSERT(irrep >= 0 && irrep < 8);

        (*this)[irrep] = x;
    }

    dpd_range operator()(int irrep, const range_t<len_type>& x)
    {
        MARRAY_ASSERT(irrep >= 0 && irrep < 8);

        dpd_range ret(*this);
        ret[irrep] = x;
        return ret;
    }

    bool operator==(const dpd_range& other) const
    {
        return std::equal(begin(), end(), other.begin());
    }

    auto operator<=>(const dpd_range& other) const
    {
        auto is_empty = [](const auto& x) { return x.empty(); };

        auto first = [&](const dpd_range& r)
        {
            auto it = std::find_if_not(r.begin(), r.end(), is_empty);
            return dpd_index(it - r.begin(), it != r.end() ? it->front() : 0);
        };

        auto last = [&](const dpd_range& r)
        {
            auto it = std::find_if_not(r.rbegin(), r.rend(), is_empty);
            return dpd_index(r.size() - (it - r.rbegin()) - 1,
                             it != r.rend() ? it->back() : 0);
        };

        if (*this == other)
            return std::partial_ordering::equivalent;

        if (last(*this) < first(other))
            return std::partial_ordering::less;

        if (last(other) < first(*this))
            return std::partial_ordering::greater;

        return std::partial_ordering::unordered;
    }
};

namespace detail
{

template <typename T>
struct is_dpd_index_or_slice_helper
: std::bool_constant<std::is_convertible_v<T, dpd_index>
                     || std::is_convertible_v<T, dpd_range>
                     || std::is_same_v<T, all_t>>
{
};

template <typename T>
struct is_dpd_index_or_slice
: is_dpd_index_or_slice_helper<typename std::decay<T>::type>
{
};

template <typename... Args> struct are_dpd_indices_or_slices;

template <> struct are_dpd_indices_or_slices<> : std::true_type
{
};

template <typename Arg, typename... Args>
struct are_dpd_indices_or_slices<Arg, Args...>
: std::conditional_t<is_dpd_index_or_slice<Arg>::value,
                     are_dpd_indices_or_slices<Args...>,
                     std::false_type>
{
};

template <typename... Args> struct sliced_dimension;

template <> struct sliced_dimension<>
{
    static constexpr int value = 0;
};

template <typename Arg, typename... Args> struct sliced_dimension<Arg, Args...>
{
    static constexpr int value =
        !std::is_same<std::decay_t<Arg>, dpd_index>::value
        + sliced_dimension<Args...>::value;
};

} // namespace detail

MARRAY_END_NAMESPACE

#endif // MARRAY_DPD_RANGE_HPP
