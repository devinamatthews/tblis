#ifndef _MARRAY_DPD_VARRAY_VIEW_HPP_
#define _MARRAY_DPD_VARRAY_VIEW_HPP_

#include "dpd_varray_base.hpp"

namespace MArray
{

template <typename Type>
class dpd_varray_view : public dpd_varray_base<Type, dpd_varray_view<Type>, false>
{
    protected:
        typedef dpd_varray_base<Type, dpd_varray_view, false> base;

        using base::size_;
        using base::len_;
        using base::off_;
        using base::leaf_;
        using base::parent_;
        using base::perm_;
        using base::depth_;
        using base::data_;
        using base::irrep_;
        using base::nirrep_;
        using base::layout_;

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

        dpd_varray_view() {}

        dpd_varray_view(const dpd_varray_view& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::cptr,pointer>>
        dpd_varray_view(const dpd_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::pointer,pointer>>
        dpd_varray_view(dpd_varray_base<U, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::pointer,pointer>>
        dpd_varray_view(dpd_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        dpd_varray_view(unsigned irrep, unsigned nirrep,
                        const detail::array_2d<len_type>& len, pointer ptr,
                        dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, layout);
        }

        dpd_varray_view(unsigned irrep, unsigned nirrep,
                        const detail::array_2d<len_type>& len, pointer ptr,
                        const detail::array_1d<unsigned>& depth, layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, depth, layout);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        dpd_varray_view& operator=(const dpd_varray_view& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::reset;
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
         * Mutating permute
         *
         **********************************************************************/

        void permute(const detail::array_1d<unsigned>& perm_)
        {
            unsigned ndim = dimension();

            MARRAY_ASSERT(perm_.size() == ndim);

            dim_vector new_perm(ndim);
            dim_vector perm;
            perm_.slurp(perm);

            for (unsigned i = 0;i < perm.size();i++)
                new_perm[i] = this->perm_[perm[i]];

            this->perm_.swap(new_perm);
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

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(dpd_varray_view& other)
        {
            base::swap(other);
        }

        friend void swap(dpd_varray_view& a, dpd_varray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
