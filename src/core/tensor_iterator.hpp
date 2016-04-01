#ifndef _TBLIS_ITERATOR_HPP_
#define _TBLIS_ITERATOR_HPP_

#include "tblis.hpp"

#include "util/util.hpp"

#include <vector>
#include <iostream>
#include <type_traits>
#include <array>

namespace tblis
{

namespace detail
{
    template <typename... Ts>
    struct are_inc_t_vectors_helper;

    template <>
    struct are_inc_t_vectors_helper<> : std::true_type {};

    template <typename T, typename... Ts>
    struct are_inc_t_vectors_helper<T, Ts...>
    : std::integral_constant<bool,std::is_same<T,std::vector<inc_t>>::value &&
                                  are_inc_t_vectors_helper<Ts...>::value> {};

    template <gint_t N, typename... Ts>
    struct are_inc_t_vectors
    : std::integral_constant<bool,(N == sizeof...(Ts)) &&
                                  are_inc_t_vectors_helper<Ts...>::value> {};

    template <size_t N, size_t I, typename Offset, typename... Offsets>
    struct inc_offsets_helper;

    template <size_t N, typename Offset>
    struct inc_offsets_helper<N, N, Offset>
    {
        inc_offsets_helper(int i, Offset& off0, const std::array<std::vector<inc_t>,N>& strides)
        {
            off0 += strides[N-1][i];
        }
    };

    template <size_t N, size_t I, typename Offset, typename... Offsets>
    struct inc_offsets_helper
    {
        inc_offsets_helper(gint_t i, Offset& off0, Offsets&... off, const std::array<std::vector<inc_t>,N>& strides)
        {
            off0 += strides[I-1][i];
            inc_offsets_helper<N,I+1,Offsets...>(i, off..., strides);
        }
    };

    template <size_t N, typename... Offsets>
    void inc_offsets(gint_t i, const std::array<std::vector<inc_t>,N>& strides, Offsets&... off)
    {
        inc_offsets_helper<N,1,Offsets...>(i, off..., strides);
    }

    template <size_t N, size_t I, typename Offset, typename... Offsets>
    struct dec_offsets_helper;

    template <size_t N, typename Offset>
    struct dec_offsets_helper<N, N, Offset>
    {
        dec_offsets_helper(gint_t i, Offset& off0, const std::vector<inc_t>& pos, const std::array<std::vector<inc_t>,N>& strides)
        {
            off0 -= pos[i]*strides[N-1][i];
        }
    };

    template <size_t N, size_t I, typename Offset, typename... Offsets>
    struct dec_offsets_helper
    {
        dec_offsets_helper(gint_t i, Offset& off0, Offsets&... off, const std::vector<inc_t>& pos, const std::array<std::vector<inc_t>,N>& strides)
        {
            off0 -= pos[i]*strides[I-1][i];
            dec_offsets_helper<N,I+1,Offsets...>(i, off..., pos, strides);
        }
    };

    template <size_t N, typename... Offsets>
    void dec_offsets(gint_t i, const std::vector<inc_t>& pos, const std::array<std::vector<inc_t>,N>& strides, Offsets&... off)
    {
        dec_offsets_helper<N,1,Offsets...>(i, off..., pos, strides);
    }

    template <size_t N, size_t I, typename Offset, typename... Offsets>
    struct set_offsets_helper;

    template <size_t N, typename Offset>
    struct set_offsets_helper<N, N, Offset>
    {
        set_offsets_helper(Offset& off0, const std::vector<inc_t>& pos, const std::array<std::vector<inc_t>,N>& strides)
        {
            off0 = 0;
            for (size_t i = 0;i < pos.size();i++) off0 += pos[i]*strides[N-1][i];
        }
    };

    template <size_t N, size_t I, typename Offset, typename... Offsets>
    struct set_offsets_helper
    {
        set_offsets_helper(Offset& off0, Offsets&... off, const std::vector<inc_t>& pos, const std::array<std::vector<inc_t>,N>& strides)
        {
            off0 = 0;
            for (size_t i = 0;i < pos.size();i++) off0 += pos[i]*strides[I-1][i];
            set_offsets_helper<N,I+1,Offsets...>(off..., pos, strides);
        }
    };

    template <size_t N, typename... Offsets>
    void set_offsets(const std::vector<inc_t>& pos, const std::array<std::vector<inc_t>,N>& strides, Offsets&... off)
    {
        set_offsets_helper<N,1,Offsets...>(off..., pos, strides);
    }
}

template <int N=1>
class Iterator
{
    public:
        Iterator() : _first(true) {}

        Iterator(const Iterator&) = default;

        Iterator(Iterator&&) = default;

        template <typename... Strides,
                  typename=typename std::enable_if<sizeof...(Strides) == N>::type>
        Iterator(const std::vector<dim_t>& len, const Strides&... strides)
        : _pos(len.size()), _len(len), _strides{strides...}, _first(true)
        {
            check();
        }

        Iterator& operator=(const Iterator&) = default;

        Iterator& operator=(Iterator&&) = default;

        void reset()
        {
            _pos.clear();
            _len.clear();
            for (int i = 0;i < N;i++) _strides[i].clear();
            _first = true;
        }

        template <typename... Strides,
                  typename=typename std::enable_if<sizeof...(Strides) == N>::type>
        void reset(const std::vector<dim_t>& len, const Strides&... strides)
        {
            _pos.assign(len.size(), 0);
            _len = len;
            _strides = {strides...};
            _first = true;

            check();
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        bool next(Offsets&... off)
        {
            if (_first)
            {
                _first = false;
            }
            else
            {
                if (_len.size() == 0)
                {
                    _first = true;
                    return false;
                }

                for (gint_t i = 0;i < _len.size();i++)
                {
                    if (_pos[i] == _len[i]-1)
                    {
                        detail::dec_offsets(i, _pos, _strides, off...);
                        _pos[i] = 0;

                        if (i == _len.size()-1)
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
            }

            return true;
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        void position(dim_t pos, Offsets&... off)
        {
            for (size_t i = 0;i < _len.size();i++)
            {
                _pos[i] = pos%_len[i];
                pos = pos/_len[i];
            }
            ASSERT(pos == 0);

            position(_pos, off...);
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        void position(const std::vector<dim_t>& pos, Offsets&... off)
        {
            ASSERT(pos.size() == _pos.size());

            _pos = pos;

            for (size_t i = 0;i < _pos.size();i++)
            {
                ASSERT(_pos[i] >= 0 && _pos[i] < _len[i]);
            }

            detail::set_offsets(_pos, _strides, off...);

            _first = true;
        }

        gint_t dimension() const
        {
            return _len.size();
        }

        dim_t position(gint_t i) const
        {
            return _pos[i];
        }

        const std::vector<dim_t>& position() const
        {
            return _pos;
        }

        dim_t length(gint_t i) const
        {
            return _len[i];
        }

        const std::vector<dim_t>& lengths() const
        {
            return _len;
        }

        template <int I=0>
        dim_t stride(gint_t i) const
        {
            return _strides[I][i];
        }

        template <int I=0>
        const std::vector<dim_t>& strides() const
        {
            return _strides[I];
        }

        friend void swap(Iterator& i1, Iterator& i2)
        {
            using std::swap;
            swap(i1._pos, i2._pos);
            swap(i1._len, i2._len);
            swap(i1._strides, i2._strides);
            swap(i1._first, i2._first);
        }

    private:
        void check()
        {
            for (auto& s : _strides)
            {
                ASSERT(s.size() == _len.size());
            }

            for (dim_t l : _len)
            {
                ASSERT(l > 0);
            }
        }

        std::vector<dim_t> _pos;
        std::vector<dim_t> _len;
        std::array<std::vector<inc_t>,N> _strides;
        bool _first;
};

}

#endif
