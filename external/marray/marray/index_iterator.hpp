#ifndef MARRAY_INDEX_ITERATOR_HPP
#define MARRAY_INDEX_ITERATOR_HPP

#include <algorithm>

#include "range.hpp"

namespace MArray
{

namespace detail
{

inline void check_size(size_t) {}

template <typename Stride, typename... Strides>
void check_size(size_t size, const Stride& stride, const Strides&... strides)
{
    MARRAY_ASSERT(size == stride.size());
    check_size(size, strides...);
}

}

/**
 * An iterator over a set of indices, such as those which index elements in a tensor.
 *
 * Optionally, one or more sets of strides can be supplied so that pointers or pointer-like
 * offsets can be tracked along with the indices. Note that this class should not be constructed
 * directly, use @ref make_iterator instead.
 *
 * @tparam NDim     Either a non-negative number of dimensions (indices), or [DYNAMIC](@ref MArray::DYNAMIC).
 *
 * @tparam N        The number of sets of strides to track.
 */
template <int NDim, int N>
class index_iterator
{
    public:
        /**
         * Construct an iterator given a set of lengths and zero or more sets of strides.
         *
         * The order of iteration is optimized for the first set of strides, if given.
         *
         * @param len       The length of each index. The number of lengths must match `NDim` if
         *                  it is not [DYNAMIC](@ref MArray::DYNAMIC). Can be any container type with values convertible
         *                  to [len_type](@ref MArray::len_type).
         *
         * @param strides   A parameter pack of index strides. There must be exactly `N` parameters
         *                  in the pack, and each set of strides must have the same size as `len`.
         *                  Can be any container type with values convertible to @ref stride_type.
         */
#if !MARRAY_DOXYGEN
        template <typename Len, typename... Strides,
                  typename=std::enable_if_t<detail::is_container_of<Len, len_type>::value &&
                                            detail::are_containers_of<stride_type, Strides...>::value &&
                                            sizeof...(Strides) == N>>
#endif
        index_iterator(const Len& len, const Strides&... strides)
        : ndim_(len.size()), pos_(ndim_), first_(true), empty_(false)
        {
            MARRAY_ASSERT(NDim == DYNAMIC || NDim == len.size());

            detail::check_size(ndim_, strides...);
            detail::assign(len_, len);
            detail::assign(strides_, strides...);

            for (auto i : range(dimension()))
                if (length(i) == 0)
                    empty_ = true;
        }

        /**
         * Set the iterator back to its starting state (all indices 0).
         */
        void reset()
        {
            std::fill(pos_.begin(), pos_.end(), 0);
            first_ = true;
        }

        /**
         * Set the values of the indices to the "next" combination, in lexicographical order, and return
         * true if such a combination exists and false otherwise.
         *
         * @note Unlike `std::next_permutation`, this function is intended to be called *before* using the indices,
         * and not afterwards. Example:
         *
         * @code{.cxx}
         * auto it = make_iterator(lengths, strides);
         * auto off = 0;
         * while (it.next(off))
         * {
         *     ...
         * }
         * @endcode
         *
         * @note An iterator with 0 dimensions technically has no valid index combinations, but
         * `next` will return true exactly once.
         *
         * @code{.cxx}
         * std::vector<int> len;
         * auto it = make_iterator(len);
         * while (it.next())
         * {
         *     // will execute exactly once
         *     ...
         * }
         * @endcode
         *
         * @note After `next` returns false (i.e. the valid index combinations)
         * have been exhausted), the iterator is returned to its initial state (as if `reset` were called) and
         * can be used again. Example:
         *
         * @code{.cxx}
         * auto outer = make_iterator(...);
         * auto inner = make_iterator(...);
         * auto off = 0;
         * while (outer.next(off))
         * {
         *     while (inner.next(off))
         *     {
         *         ...
         *     }
         * }
         * @endcode
         *
         * @param off   A parameter pack of offsets. There must be exactly `N` parameters
         *              in the pack, and each offset must be capable of modification via
         *              `offset += increment`. At all times, the difference between each offset
         *              and its initial value (user-defined) is equal to the sum of the value of all
         *              current indices multiplied by their respective strides.
         *
         * @returns     True if the current indices represent a new valid combination, and false otherwise,
         *              indicating all valid combinations have been exhausted.
         */
#if !MARRAY_DOXYGEN
        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
#endif
        bool next(Offsets&... off)
        {
            if (empty_) return false;

            if (first_)
            {
                first_ = false;
                return true;
            }

            if (ndim_ == 0)
            {
                first_ = true;
                return false;
            }

            for (auto i : range(ndim_))
            {
                if (pos_[i] == len_[i]-1)
                {
                    detail::dec_offsets(i, pos_, strides_, off...);
                    pos_[i] = 0;

                    if (i == ndim_-1)
                    {
                        first_ = true;
                        return false;
                    }
                }
                else
                {
                    detail::inc_offsets(i, strides_, off...);
                    pos_[i]++;
                    return true;
                }
            }

            return true;
        }

        /**
         * Reposition the iterator based on a linear index. This sets the iterator to the state *just before*
         * the index combination denoted by `pos`. This means that the user is expected to call @ref next before
         * using the iterator or any computed offsets. Example:
         *
         * @code{.cxx}
         * auto it = make_iterator(lengths, strides);
         * auto off = 0;
         * it.position(10, off);
         * while (it.next(off))
         * {
         *     // this loop will visit all combinations except the first 10, and the values of `off` will be
         *     // exactly the same as if we did not call `position`
         *     ...
         * }
         *
         * off = 0;
         * it.position(10, off);
         * for (auto i = 10;i < 30 && it.next(off);i++)
         * {
         *     // this loop will visit combinations 10 through 30, if they exist
         *     ...
         * }
         * @endcode
         *
         * @param pos   The desired index combination, in the order in which they would normally be visitied by
         *              the iterator. If the specified combination does not exist, the iterator is left unchanged.
         *              0 is the only valid value for an iterator with 0 dimensions.
         *
         * @param off   A parameter pack of offsets. There must be exactly `N` parameters in the pack, and each
         *              offset must be capable of modification via `offset += increment`. The value of each offset is
         *              assumed to be the initial (user-defined) value, such as for a newly-created iterator, and
         *              will be updated to reflect to new index combination.
         *
         * @returns     True if the indicated indicated combination exists, and false otherwise.
         */
#if !MARRAY_DOXYGEN
        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
#endif
        bool position(stride_type pos, Offsets&... off)
        {
            if (empty_)
            {
                if (pos == 0)
                {
                    first_ = true;
                    return true;
                }
                else
                {
                    return false;
                }
            }

            if (pos < 0)
                return false;

            for (auto i : range(ndim_))
            {
                pos_[i] = pos%len_[i];
                pos = pos/len_[i];
            }

            if (pos > 0)
                return false;

            return position(pos_, off...);
        }

        /**
         * Reposition the iterator based on a given index combination. This sets the iterator to the state *just before*
         * the index combination denoted by `pos`. This means that the user is expected to call @ref next before
         * using the iterator or any computed offsets. Example:
         *
         * @code{.cxx}
         * auto it = make_iterator(lengths, strides);
         * auto off = 0;
         * it.position(pos, off);
         * while (it.next(off))
         * {
         *     // this loop will visit all combinations starting at `pos`, and the values of `off` will be
         *     // exactly the same as if we did not call `position`
         *     ...
         * }
         * @endcode
         *
         * @param pos   The desired index combination. If the specified combination does not exist, the iterator
         *              is left unchanged. Can be any container type with values convertible to [len_type](@ref MArray::len_type).
         *              `pos` must be empty for an iterator with 0 dimensions.
         *
         * @param off   A parameter pack of offsets. There must be exactly `N` parameters in the pack, and each
         *              offset must be capable of modification via `offset += increment`. The value of each offset is
         *              assumed to be the initial (user-defined) value, such as for a newly-created iterator, and
         *              will be updated to reflect to new index combination.
         *
         * @returns     True if the indicated indicated combination exists, and false otherwise.
         */
#if !MARRAY_DOXYGEN
        template <typename Pos, typename... Offsets,
                  typename=typename std::enable_if<detail::is_container_of<Pos, len_type>::value &&
                                                   sizeof...(Offsets) == N>::type>
#endif
        bool position(const Pos& pos, Offsets&... off)
        {
            MARRAY_ASSERT(pos.size() == ndim_);

            auto it = pos.begin();
            for (auto i : range(ndim_))
            {
                if (*it < 0 || *it >= len_[i])
                    return false;
                ++it;
            }

            if (empty_)
                return true;

            first_ = true;
            std::copy_n(pos.begin(), ndim_, pos_.begin());
            detail::move_offsets(pos_, strides_, off...);

            return true;
        }

        /**
         * Reorder the dimensions such that iteration is optimized for the given set of strides. Optimized ordering
         * ensures that the corresponding offsets are strictly non-decreasing during iteration.
         *
         * @param which     The set of strides for which to optimize ordering. Must be non-negative and less than `N`.
         *
         * @returns    The permutation which should be applied to the position (@ref position()) in order to reproduce
         *             the previous ordering.
         */
        detail::array_type_t<int,NDim> optimize_for(int which)
        {
            MARRAY_ASSERT(which >= 0 && which < N);

            if (dimension() < 2 || empty_) return;

            detail::array_type_t<int,NDim> perm(dimension());
            for (auto i : range(dimension()))
                perm[i] = i;

            /*
             * The most common scenarios are a) already sorted as desired (column major),
             * or b) in reverse sorted order (row major). If we suspect the latter case,
             * go ahead and reverse so that we can finish the sort in less than O(n^2) time.
             */
            if (strides_[which][0] > strides_[which][ndim_-1])
            {
                std::reverse(len_.begin(), len_.end());
                std::reverse(perm.begin(), perm.end());
                for (auto i : range(N))
                    std::reverse(strides_[i].begin(), strides_[i].end());
            }

            /*
             * Insertion sort of lengths + all strides, using chosen strides as the key.
             */
            for (auto i : range(1, ndim_))
            for (auto j = i;j > 0 && strides_[which][j] < strides_[which][j-1];j--)
            {
                using std::swap;
                swap(len_[j-1], len_[j]);
                swap(perm[j-1], perm[j]);
                for (auto k : range(N))
                    swap(strides_[k][j-1], strides_[k][j]);
            }

            detail::array_type_t<int,NDim> iperm(dimension());
            for (auto i : range(dimension()))
                iperm[perm[i]] = i;

            return iperm;
        }

        /**
         * Return the number of dimensions (indices).
         *
         * @returns     The number of dimensions (indices).
         */
        auto dimension() const
        {
            return ndim_;
        }

        /**
         * Return the current index combination.
         *
         * @returns     The current index combination.
         */
        auto& position() const
        {
            return pos_;
        }

        /**
         * Return the length of the given dimension.
         *
         * @param dim   The dimension.
         *
         * @returns     The length.
         */
        auto length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < ndim_);
            return len_[dim];
        }

        /**
         * Return the lengths of all dimensions.
         *
         * @returns     The lengths of all dimensions.
         */
        auto& lengths() const
        {
            return len_;
        }

        /**
         * Return one of the strides along the given dimension.
         *
         * @param i     The set of strides, 0 <= `i` < `N`.
         *
         * @param dim   The dimension.
         *
         * @returns     The stride.
         */
        auto stride(int i, int dim) const
        {
            MARRAY_ASSERT(i >= 0 && i < N);
            MARRAY_ASSERT(dim >= 0 && dim < ndim_);
            return strides_[i][dim];
        }

        /**
         * Returns one of the sets of strides.
         *
         * @param i     The set of strides, 0 <= `i` < `N`.
         *
         * @returns     The strides along each dimension.
         */
        auto& strides(int i) const
        {
            MARRAY_ASSERT(i >= 0 && i < N);
            return strides_[i];
        }

        /**
         * Swap two iterators.
         *
         * @param other     The iterator to swap with.
         */
        void swap(index_iterator& other)
        {
            using std::swap;
            swap(ndim_, other.ndim_);
            swap(pos_, other.pos_);
            swap(len_, other.len_);
            swap(strides_, other.strides_);
            swap(first_, other.first_);
            swap(empty_, other.empty_);
        }

        /**
         * Swap two iterators.
         *
         * @param lhs     An iterator.
         *
         * @param rhs     Another iterator.
         */
        friend void swap(index_iterator& lhs, index_iterator& rhs)
        {
            lhs.swap(rhs);
        }

        /**
         * Check if the iteration space is empty.
         *
         * @returns True if there are no valid index combinations.
         */
        bool empty() const
        {
            return empty_;
        }

    private:
        size_t ndim_ = 0;
        detail::array_type_t<len_type,NDim> pos_;
        detail::array_type_t<len_type,NDim> len_;
        std::array<detail::array_type_t<stride_type,NDim>,N> strides_;
        bool first_;
        bool empty_;
};

/**
 * Create an @ref index_iterator from the given set of lengths and zero or more sets of strides.
 *
 * @code{.cxx}
 * std::vector<int> lengths{4, 5, 2};
 * std::vector<int> row_strides{10, 2, 1};
 * std::vector<int> col_strides{1, 4, 20};
 *
 * auto it = make_iterator(lengths, row_strides, col_strides);
 * auto row_off = 0, col_off = 0;
 *
 * while (it.next(row_off, col_off))
 * {
 *     ...
 * }
 * @endcode
 *
 * @param len       The lengths of each dimension. Can be any container type whose values are convertible to
 *                  [len_type](@ref MArray::len_type).
 *
 * @param strides   A parameter pack of strides. There must be exactly `N` parameters in the pack. Can be any
 *                  container type whose values are convertible to @ref stride_type. The size of each set of strides
 *                  must be the same as the size of `len`.
 *
 * @returns         An @ref index_iterator.
 *
 * @ingroup funcs
 */
#if !MARRAY_DOXYGEN
template <typename Lengths, typename... Strides,
          typename=std::enable_if_t<detail::is_container_of<Lengths, len_type>::value &&
                                    detail::are_containers_of<stride_type, Strides...>::value>>
index_iterator<detail::container_size<Lengths>::value, sizeof...(Strides)>
#else
index_iterator
#endif
make_iterator(const Lengths& len,
              const Strides&... strides)
{
    return {len, strides...};
}

}

#endif //MARRAY_INDEX_ITERATOR_HPP
