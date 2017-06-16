#ifndef _MARRAY_DPD_MARRAY_VIEW_HPP_
#define _MARRAY_DPD_MARRAY_VIEW_HPP_

#include "dpd_marray_base.hpp"

namespace MArray
{

template <typename Type, unsigned NDim>
class dpd_marray_view : public dpd_marray_base<Type, NDim, dpd_marray_view<Type, NDim>, false>
{
    protected:
        typedef dpd_marray_base<Type, NDim, dpd_marray_view, false> base;

        using base::len_;
        using base::size_;
        using base::perm_;
        using base::data_;
        using base::irrep_;
        using base::nirrep_;
        using base::layout_;

        template <typename U> using initializer_matrix =
            std::initializer_list<std::initializer_list<U>>;

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

        dpd_marray_view() {}

        dpd_marray_view(const dpd_marray_view& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename dpd_marray_base<U, NDim, D, O>::cptr,pointer>>
        dpd_marray_view(const dpd_marray_base<U, NDim, D, O>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename dpd_marray_base<U, NDim, D, O>::pointer,pointer>>
        dpd_marray_view(dpd_marray_base<U, NDim, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename dpd_marray_base<U, NDim, D, O>::pointer,pointer>>
        dpd_marray_view(dpd_marray_base<U, NDim, D, O>& other)
        {
            reset(other);
        }

        template <typename U, typename=
            detail::enable_if_assignable_t<len_type&,U>>
        dpd_marray_view(unsigned irrep, unsigned nirrep,
                        initializer_matrix<U> len, pointer ptr,
                        dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, layout);
        }

        template <typename U, typename=
            detail::enable_if_container_of_t<U,len_type>>
        dpd_marray_view(unsigned irrep, unsigned nirrep,
                        std::initializer_list<U> len, pointer ptr,
                        dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, layout);
        }

        template <typename U, typename=
            detail::enable_if_container_of_containers_of_t<U,len_type>>
        dpd_marray_view(unsigned irrep, unsigned nirrep, const U& len, pointer ptr,
                        dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, layout);
        }

        template <typename U>
        dpd_marray_view(unsigned irrep, unsigned nirrep,
                        matrix_view<U> len, pointer ptr,
                        dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, layout);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        dpd_marray_view& operator=(const dpd_marray_view& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::reset;
        using base::cview;
        using base::view;
        using base::permuted;
        using base::transposed;
        using base::T;
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

        void permute(std::initializer_list<unsigned> perm)
        {
            permute<>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        void permute(const U& perm)
        {
            std::array<unsigned, NDim> new_perm;

            auto it = perm.begin();
            for (unsigned i = 0;i < NDim;i++)
            {
                new_perm[i] = perm_[*it];
                ++it;
            }

            perm_ = new_perm;
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        void transpose()
        {
            permute({1, 0});
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

        void swap(dpd_marray_view& other)
        {
            base::swap(other);
        }

        friend void swap(dpd_marray_view& a, dpd_marray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
