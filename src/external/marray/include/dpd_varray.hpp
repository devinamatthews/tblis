#ifndef _MARRAY_DPD_VARRAY_HPP_
#define _MARRAY_DPD_VARRAY_HPP_

#include "dpd_varray_view.hpp"

namespace MArray
{

template <typename Type, typename Allocator>
class dpd_varray : public dpd_varray_base<Type, dpd_varray<Type, Allocator>, true>
{
    protected:
        typedef dpd_varray_base<Type, dpd_varray, true> base;
        typedef std::allocator_traits<Allocator> alloc_traits;

        using base::size_;
        using base::len_;
        using base::off_;
        using base::leaf_;
        using base::parent_;
        using base::perm_;
        using base::data_;
        using base::irrep_;
        using base::nirrep_;
        struct : Allocator { stride_type size = 0; } storage_;

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

        dpd_varray() {}

        dpd_varray(const dpd_varray& other)
        {
            reset(other);
        }

        dpd_varray(dpd_varray&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        dpd_varray(const dpd_varray<U, A>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        dpd_varray(const dpd_varray_base<U, D, O>& other, dpd_layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        dpd_varray(const dpd_varray_base<U, D, O>& other,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset(other, depth, layout);
        }

        dpd_varray(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const Type& val = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, val, layout);
        }

        dpd_varray(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, const Type& val,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, val, depth, layout);
        }

        dpd_varray(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, dpd_layout layout)
        {
            reset(irrep, nirrep, len, Type(), layout);
        }

        dpd_varray(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& depth, layout layout)
        {
            reset(irrep, nirrep, len, Type(), depth, layout);
        }

        dpd_varray(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, uninitialized_t,
                   dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, uninitialized, layout);
        }

        dpd_varray(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, uninitialized_t,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, uninitialized, depth, layout);
        }

        ~dpd_varray()
        {
            reset();
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        dpd_varray& operator=(const dpd_varray& other)
        {
            return base::operator=(other);
        }

        stride_type size() const
        {
            return size_[2*dimension()-2][irrep_];
        }

        using base::operator=;
        using base::cview;
        using base::view;
        using base::permuted;
        using base::operator();
        using base::cdata;
        using base::data;
        using base::length;
        using base::lengths;
        using base::irrep;
        using base::num_irreps;
        using base::permutation;
        using base::dimension;
        using base::size;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            if (data_)
            {
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_+i);
                }
                alloc_traits::deallocate(storage_, data_, storage_.size);
                storage_.size = 0;
            }

            base::reset();
        }

        void reset(dpd_varray&& other)
        {
            swap(other);
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const dpd_varray<U, A>& other)
        {
            reset();
            base::reset(const_cast<dpd_varray<U, A>&>(other));

            storage_.size = other.size();
            data_ = alloc_traits::allocate(storage_, storage_.size);
            *this = other;
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const dpd_varray_base<U, D, O>& other, dpd_layout layout = DEFAULT)
        {
            auto len = other.lengths();

            if (std::is_scalar<Type>::value)
            {
                reset(other.irrep_, other.nirrep_, len.view(), uninitialized, layout);
            }
            else
            {
                reset(other.irrep_, other.nirrep_, len.view(), layout);
            }

            *this = other;
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const dpd_varray_base<U, D, O>& other,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            auto len = other.lengths();

            if (std::is_scalar<Type>::value)
            {
                reset(other.irrep_, other.nirrep_, len.view(), uninitialized, depth, layout);
            }
            else
            {
                reset(other.irrep_, other.nirrep_, len.view(), depth, layout);
            }

            *this = other;
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const Type& val = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, uninitialized, layout);
            std::uninitialized_fill_n(data_, storage_.size, val);
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, const Type& val,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, uninitialized, depth, layout);
            std::uninitialized_fill_n(data_, storage_.size, val);
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, dpd_layout layout)
        {
            reset(irrep, nirrep, len, Type(), layout);
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& depth, layout layout)
        {
            reset(irrep, nirrep, len, Type(), depth, layout);
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, uninitialized_t,
                   dpd_layout layout = DEFAULT)
        {
            reset();

            storage_.size = size(irrep, len);
            base::reset(irrep, nirrep, len,
                        alloc_traits::allocate(storage_, storage_.size),
                        layout);
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, uninitialized_t,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset();

            storage_.size = size(irrep, len);
            base::reset(irrep, nirrep, len,
                        alloc_traits::allocate(storage_, storage_.size),
                        depth, layout);
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(dpd_varray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            base::swap(other);
        }

        friend void swap(dpd_varray& a, dpd_varray& b)
        {
            a.swap(b);
        }
};

}

#endif
