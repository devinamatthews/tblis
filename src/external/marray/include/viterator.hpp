#ifndef _MARRAY_VITERATOR_HPP_
#define _MARRAY_VITERATOR_HPP_

#include "utility.hpp"

namespace MArray
{

template <unsigned N=1>
class viterator
{
    public:
        viterator() {}

        viterator(const viterator&) = default;

        viterator(viterator&&) = default;

        template <typename Len, typename... Strides,
                  typename=typename std::enable_if<detail::is_container<Len>::value &&
                                                   detail::are_containers<Strides...>::value &&
                                                   sizeof...(Strides) == N>::type>
        viterator(const Len& len, const Strides&... strides)
        : ndim_(len.size()), pos_(len.size()), len_(len.size()), first_(true), empty_(false)
        {
            for (unsigned i = 0;i < ndim_;i++) if (len[i] == 0) empty_ = true;
            std::copy_n(len.begin(), ndim_, len_.begin());
            for (unsigned i = 0;i < N;i++) strides_[i].resize(len.size());
            detail::set_strides(strides_, strides...);
        }

        viterator& operator=(const viterator&) = default;

        viterator& operator=(viterator&&) = default;

        void reset()
        {
            pos_.assign(ndim_, 0);
            first_ = true;
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
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

            for (unsigned i = 0;i < ndim_;i++)
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

        void prev()
        {
            if (empty_ || ndim_ == 0) return;

            for (unsigned i = 0;i < ndim_;i++)
            {
                if (pos_[i] == 0)
                {
                    pos_[i] = len_[i]-1;
                }
                else
                {
                    pos_[i]--;
                    return;
                }
            }
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        void position(stride_type pos, Offsets&... off)
        {
            if (empty_) return;

            for (size_t i = 0;i < ndim_;i++)
            {
                pos_[i] = pos%len_[i];
                pos = pos/len_[i];
            }
            MARRAY_ASSERT(pos == 0);

            position(pos_, off...);
        }

        template <typename Pos, typename... Offsets,
                  typename=typename std::enable_if<detail::is_container_of<Pos, len_type>::value &&
                                                   sizeof...(Offsets) == N>::type>
        void position(const Pos& pos, Offsets&... off)
        {
            if (empty_) return;

            MARRAY_ASSERT(pos.size() == ndim_);

            pos_.assign(pos.begin(), pos.end());

            for (size_t i = 0;i < ndim_;i++)
            {
                MARRAY_ASSERT(pos_[i] >= 0 && pos_[i] < len_[i]);
            }

            detail::move_offsets(pos_, strides_, off...);

            first_ = true;
        }

        unsigned dimension() const
        {
            return ndim_;
        }

        const len_vector& position() const
        {
            return pos_;
        }

        len_type length(unsigned dim) const
        {
            return len_[dim];
        }

        const len_vector& lengths() const
        {
            return len_;
        }

        stride_type stride(unsigned i, unsigned dim) const
        {
            return strides_[i][dim];
        }

        const stride_vector& strides(unsigned i) const
        {
            return strides_[i];
        }

        void swap(viterator& i2)
        {
            using std::swap;
            swap(ndim_, i2.ndim_);
            swap(pos_, i2.pos_);
            swap(len_, i2.len_);
            swap(strides_, i2.strides_);
            swap(first_, i2.first_);
            swap(empty_, i2.empty_);
        }

        friend void swap(viterator& i1, viterator& i2)
        {
            i1.swap(i2);
        }

    private:
        size_t ndim_ = 0;
        len_vector pos_;
        len_vector len_;
        std::array<stride_vector,N> strides_;
        bool first_ = true;
        bool empty_ = true;
};

template <typename Length, typename... Strides,
          typename=detail::enable_if_t<detail::are_containers_of<stride_type, Strides...>::value>>
viterator<sizeof...(Strides)>
make_iterator(const std::vector<Length>& len,
              const Strides&... strides)
{
    return {len, strides...};
}

template <typename... Strides,
          typename=detail::enable_if_t<detail::are_containers_of<stride_type, Strides...>::value>>
viterator<sizeof...(Strides)>
make_iterator(const len_vector& len,
              const Strides&... strides)
{
    return {len, strides...};
}

}

#endif
