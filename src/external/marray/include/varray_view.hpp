#ifndef _MARRAY_VARRAY_VIEW_HPP_
#define _MARRAY_VARRAY_VIEW_HPP_

#include "varray_base.hpp"

namespace MArray
{

template <typename Type>
class varray_view : public varray_base<Type, varray_view<Type>, false>
{
    protected:
        typedef varray_base<Type, varray_view, false> base;

        using base::len_;
        using base::stride_;
        using base::data_;

    public:
        using typename base::value_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        varray_view() {}

        varray_view(const varray_view& other)
        {
            reset(other);
        }

        varray_view(varray_view&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename varray_base<U, D, O>::cptr,pointer>>
        varray_view(const varray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename varray_base<U, D, O>::pointer,pointer>>
        varray_view(varray_base<U, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename varray_base<U, D, O>::pointer,pointer>>
        varray_view(varray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, int N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::cptr,pointer>>
        varray_view(const marray_base<U, N, D, O>& other)
        {
            reset(other);
        }

        template <typename U, int N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::pointer,pointer>>
        varray_view(marray_base<U, N, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, int N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::pointer,pointer>>
        varray_view(marray_base<U, N, D, O>& other)
        {
            reset(other);
        }

        varray_view(detail::array_1d<len_type> len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }

        varray_view(detail::array_1d<len_type> len, pointer ptr, detail::array_1d<stride_type> stride)
        {
            reset(len, ptr, stride);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        varray_view& operator=(const varray_view& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::reset;
        using base::cview;
        using base::view;
        using base::fix;
        using base::shifted;
        using base::shifted_up;
        using base::shifted_down;
        using base::permuted;
        using base::lowered;
        using base::cfront;
        using base::front;
        using base::cback;
        using base::back;
        using base::operator();
        using base::cdata;
        using base::data;
        using base::length;
        using base::lengths;
        using base::stride;
        using base::strides;
        using base::dimension;
        using base::size;

        /***********************************************************************
         *
         * Mutating shift
         *
         **********************************************************************/

        void shift(detail::array_1d<len_type> n_)
        {
            len_vector n;
            n_.slurp(n);
            MARRAY_ASSERT(n.size() == dimension());
            for (auto i : range(dimension())) shift(i, n[i]);
        }

        void shift(int dim, len_type n)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            data_ += n*stride_[dim];
        }

        void shift_down(int dim)
        {
            shift(dim, len_[dim]);
        }

        void shift_up(int dim)
        {
            shift(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Mutating permutation
         *
         **********************************************************************/

        void permute(detail::array_1d<int> perm_)
        {
            dim_vector perm;
            perm_.slurp(perm);

            MARRAY_ASSERT(perm.size() == dimension());

            len_vector len(len_);
            stride_vector stride(stride_);

            for (auto i : range(dimension()))
            {
                MARRAY_ASSERT(perm[i] < dimension());
                for (auto j : range(i+1,dimension()))
                    MARRAY_ASSERT(perm[i] != perm[j]);

                len_[i] = len[perm[i]];
                stride_[i] = stride[perm[i]];
            }
        }

        /***********************************************************************
         *
         * Mutating dimension change
         *
         **********************************************************************/

        void lower(detail::array_1d<int> split_)
        {
            dim_vector split;
            split_.slurp(split);

            MARRAY_ASSERT(split.size() < dimension());

            int newdim = split.size()+1;
            for (auto i : range(newdim-1))
            {
                MARRAY_ASSERT(split[i] <= dimension());
                if (i != 0) MARRAY_ASSERT(split[i-1] <= split[i]);
            }

            len_vector len = len_;
            stride_vector stride = stride_;

            for (auto i : range(newdim))
            {
                auto begin = (i == 0 ? 0 : split[i-1]);
                auto end = (i == newdim-1 ? dimension()-1 : split[i]-1);
                if (begin > end) continue;

                if (stride[begin] < stride[end])
                {
                    len_[i] = len[end];
                    stride_[i] = stride[begin];
                    for (auto j : range(begin,end))
                    {
                        MARRAY_ASSERT(stride[j+1] == stride[j]*len[j]);
                        len_[i] *= len[j];
                    }
                }
                else
                {
                    len_[i] = len[end];
                    stride_[i] = stride[end];
                    for (auto j : range(begin,end))
                    {
                        MARRAY_ASSERT(stride[j] == stride[j+1]*len[j+1]);
                        len_[i] *= len[j];
                    }
                }
            }

            len_.resize(newdim);
            stride_.resize(newdim);
        }

        /***********************************************************************
         *
         * Mutating reversal
         *
         **********************************************************************/

        void reverse()
        {
            for (auto i : range(dimension())) reverse(i);
        }

        void reverse(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            data_ += (len_[dim]-1)*stride_[dim];
            stride_[dim] = -stride_[dim];
        }

        /***********************************************************************
         *
         * Basic setters
         *
         **********************************************************************/

        pointer data(pointer ptr)
        {
            std::swap(ptr, data_);
            return ptr;
        }

        len_type length(int dim, len_type len)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            std::swap(len, len_[dim]);
            return len;
        }

        stride_type stride(int dim, stride_type stride)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            std::swap(stride, stride_[dim]);
            return stride;
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(varray_view& other)
        {
            base::swap(other);
        }

        friend void swap(varray_view& a, varray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
