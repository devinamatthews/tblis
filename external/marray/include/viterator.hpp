#ifndef _MARRAY_VITERATOR_HPP_
#define _MARRAY_VITERATOR_HPP_

#include "miterator.hpp"

namespace MArray
{

template <unsigned N=1>
class viterator
{
    public:
        typedef unsigned idx_type;
        typedef ptrdiff_t stride_type;

        viterator() {}

        viterator(const viterator&) = default;

        viterator(viterator&&) = default;

        template <typename Len, typename... Strides,
                  typename=typename std::enable_if<detail::is_container<Len>::value &&
                                                   detail::are_containers<Strides...>::value &&
                                                   sizeof...(Strides) == N>::type>
        viterator(const Len& len, const Strides&... strides)
        : _ndim(len.size()), _pos(len.size()), _len(len.size()), _first(true), _empty(false)
        {
            for (unsigned i = 0;i < _ndim;i++) if (len[i] == 0) _empty = true;
            std::copy_n(len.begin(), _ndim, _len.begin());
            detail::set_strides(_strides, strides...);
        }

        viterator& operator=(const viterator&) = default;

        viterator& operator=(viterator&&) = default;

        void reset()
        {
            _pos.assign(_ndim, 0);
            _first = true;
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        bool next(Offsets&... off)
        {
            if (_empty) return false;

            if (_first)
            {
                _first = false;
                return true;
            }

            if (_ndim == 0)
            {
                _first = true;
                return false;
            }

            for (unsigned i = 0;i < _ndim;i++)
            {
                if (_pos[i] == _len[i]-1)
                {
                    detail::dec_offsets(i, _pos, _strides, off...);
                    _pos[i] = 0;

                    if (i == _ndim-1)
                    {
                        _first = true;
                        return false;
                    }
                }
                else
                {
                    detail::inc_offsets(i, _strides, off...);
                    _pos[i]++;
                    return true;
                }
            }

            return true;
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        void position(stride_type pos, Offsets&... off)
        {
            for (size_t i = 0;i < _ndim;i++)
            {
                _pos[i] = pos%_len[i];
                pos = pos/_len[i];
            }
            assert(pos == 0);

            position(_pos, off...);
        }

        template <typename Pos, typename... Offsets,
                  typename=typename std::enable_if<detail::is_container_of<idx_type, Pos>::value &&
                                                   sizeof...(Offsets) == N>::type>
        void position(const Pos& pos, Offsets&... off)
        {
            assert(pos.size() == _ndim);

            _pos = pos;

            for (size_t i = 0;i < _ndim;i++)
            {
                assert(_pos[i] >= 0 && _pos[i] < _len[i]);
            }

            detail::move_offsets(_pos, _strides, off...);

            _first = true;
        }

        unsigned dimension() const
        {
            return _ndim;
        }

        idx_type position(unsigned dim) const
        {
            return _pos[dim];
        }

        const std::vector<idx_type>& position() const
        {
            return _pos;
        }

        idx_type length(unsigned dim) const
        {
            return _len[dim];
        }

        const std::vector<idx_type>& lengths() const
        {
            return _len;
        }

        stride_type stride(unsigned i, unsigned dim) const
        {
            return _strides[i][dim];
        }

        const std::vector<stride_type>& strides(unsigned i) const
        {
            return _strides[i];
        }

        friend void swap(viterator& i1, viterator& i2)
        {
            using std::swap;
            swap(i1._ndim, i2._ndim);
            swap(i1._pos, i2._pos);
            swap(i1._len, i2._len);
            swap(i1._strides, i2._strides);
            swap(i1._first, i2._first);
            swap(i1._empty, i2._empty);
        }

    private:
        size_t _ndim = 0;
        std::vector<idx_type> _pos;
        std::vector<idx_type> _len;
        std::array<std::vector<stride_type>,N> _strides;
        bool _first = true;
        bool _empty = true;
};

template <typename idx_type, typename stride_type, typename... Strides,
          typename=typename std::enable_if<detail::are_containers<Strides...>::value>::type>
viterator<1+sizeof...(Strides)>
make_iterator(const std::vector<idx_type>& len,
              const std::vector<stride_type>& stride0,
              const Strides&... strides)
{
    return {len, stride0, strides...};
}

}

#endif
