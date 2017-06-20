#ifndef _MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP_
#define _MARRAY_INDEXED_DPD_VARRAY_VIEW_HPP_

#include "indexed_dpd_varray_base.hpp"

namespace MArray
{

template <typename Type>
class indexed_dpd_varray_view : public indexed_dpd_varray_base<Type, indexed_dpd_varray_view<Type>, false>
{
    template <typename, typename, bool> friend class indexed_dpd_varray_base;
    template <typename> friend class indexed_dpd_varray_view;
    template <typename, typename> friend class indexed_dpd_varray;

    protected:
        typedef indexed_dpd_varray_base<Type, indexed_dpd_varray_view, false> base;

        using base::dense_len_;
        using base::dense_size_;
        using base::idx_len_;
        using base::idx_irrep_;
        using base::perm_;
        using base::data_;
        using base::idx_;
        using base::irrep_;
        using base::dense_irrep_;
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

        indexed_dpd_varray_view()
        {
            reset();
        }

        indexed_dpd_varray_view(const indexed_dpd_varray_view& other)
        {
            reset(other);
        }

        indexed_dpd_varray_view(indexed_dpd_varray_view&& other)
        {
            reset(std::move(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
            typename indexed_dpd_varray_base<U, D, O>::cptr,pointer>>
        indexed_dpd_varray_view(const indexed_dpd_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_dpd_varray_base<U, D, O>::pointer,pointer>>
        indexed_dpd_varray_view(indexed_dpd_varray_base<U, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_dpd_varray_base<U, D, O>::pointer,pointer>>
        indexed_dpd_varray_view(indexed_dpd_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        indexed_dpd_varray_view(unsigned irrep, unsigned nirrep,
                   initializer_matrix<len_type> len, row_view<const pointer> ptr,
                   std::initializer_list<unsigned> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename=
            detail::enable_if_container_of_t<U,len_type>>
        indexed_dpd_varray_view(unsigned irrep, unsigned nirrep,
                   std::initializer_list<U> len, row_view<const pointer> ptr,
                   std::initializer_list<unsigned> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value>>
        indexed_dpd_varray_view(unsigned irrep, unsigned nirrep,
                   const U& len, row_view<const pointer> ptr,
                   const V& idx_irrep, matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        indexed_dpd_varray_view& operator=(const indexed_dpd_varray_view& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::reset;
        using base::cview;
        using base::view;
        using base::operator[];
        using base::cdata;
        using base::data;
        using base::indices;
        using base::dense_length;
        using base::dense_lengths;
        using base::indexed_length;
        using base::indexed_lengths;
        using base::length;
        using base::lengths;
        using base::indexed_irrep;
        using base::indexed_irreps;
        using base::irrep;
        using base::num_irreps;
        using base::num_indices;
        using base::permutation;
        using base::dimension;
        using base::dense_dimension;
        using base::indexed_dimension;

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_dpd_varray_view& other)
        {
            base::swap(other);
        }

        friend void swap(indexed_dpd_varray_view& a, indexed_dpd_varray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
