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

        varray_view(std::initializer_list<len_type> len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        varray_view(const U& len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }

        varray_view(std::initializer_list<len_type> len, pointer ptr,
                    std::initializer_list<stride_type> stride)
        {
            reset(len, ptr, stride);
        }

        template <typename U, typename V, typename=
                  detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                      detail::is_container_of<V,stride_type>::value>>
        varray_view(const U& len, pointer ptr, const V& stride)
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

        void shift(std::initializer_list<len_type> n)
        {
            shift<std::initializer_list<len_type>>(n);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        void shift(const U& n)
        {
            MARRAY_ASSERT(n.size() == dimension());
            for (unsigned i = 0;i < dimension();i++) shift(i, n[i]);
        }

        void shift(unsigned dim, len_type n)
        {
            MARRAY_ASSERT(dim < dimension());
            data_ += n*stride_[dim];
        }

        void shift_down(unsigned dim)
        {
            shift(dim, len_[dim]);
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Mutating permutation
         *
         **********************************************************************/

        void permute(std::initializer_list<unsigned> perm)
        {
            permute<std::initializer_list<unsigned>>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        void permute(const U& perm)
        {
            MARRAY_ASSERT(perm.size() == dimension());

            len_vector len(len_);
            stride_vector stride(stride_);

            auto it = perm.begin();
            for (unsigned i = 0;i < dimension();i++)
            {
                MARRAY_ASSERT((unsigned)*it < dimension());
                for (auto it2 = perm.begin();it2 != it;++it2)
                    MARRAY_ASSERT(*it != *it2);

                len_[i] = len[*it];
                stride_[i] = stride[*it];
                ++it;
            }
        }

        /***********************************************************************
         *
         * Mutating dimension change
         *
         **********************************************************************/

        void lower(std::initializer_list<unsigned> split)
        {
            lower<std::initializer_list<unsigned>>(split);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        void lower(const U& split_)
        {
            dim_vector split(split_.begin(), split_.end());

            MARRAY_ASSERT(split.size() < dimension());

            unsigned newdim = split.size()+1;
            for (unsigned i = 0;i < newdim-1;i++)
            {
                MARRAY_ASSERT(split[i] <= dimension());
                if (i != 0) MARRAY_ASSERT(split[i-1] <= split[i]);
            }

            len_vector len = len_;
            stride_vector stride = stride_;

            for (unsigned i = 0;i < newdim;i++)
            {
                unsigned begin = (i == 0 ? 0 : split[i-1]);
                unsigned end = (i == newdim-1 ? dimension()-1 : split[i]-1);
                if (begin > end) continue;

                if (stride[begin] < stride[end])
                {
                    len_[i] = len[end];
                    stride_[i] = stride[begin];
                    for (auto j = begin;j < end;j++)
                    {
                        MARRAY_ASSERT(stride[j+1] == stride[j]*len[j]);
                        len_[i] *= len[j];
                    }
                }
                else
                {
                    len_[i] = len[end];
                    stride_[i] = stride[end];
                    for (auto j = begin;j < end;j++)
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
            for (unsigned i = 0;i < dimension();i++) reverse(i);
        }

        void reversed(unsigned dim)
        {
            MARRAY_ASSERT(dim < dimension());
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

        len_type length(unsigned dim, len_type len)
        {
            MARRAY_ASSERT(dim < dimension());
            std::swap(len, len_[dim]);
            return len;
        }

        stride_type stride(unsigned dim, stride_type stride)
        {
            MARRAY_ASSERT(dim < dimension());
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
