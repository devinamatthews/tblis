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

        using base::size_;
        using base::len_;
        using base::off_;
        using base::stride_;
        using base::idx_irrep_;
        using base::leaf_;
        using base::parent_;
        using base::perm_;
        using base::depth_;
        using base::data_;
        using base::idx_len_;
        using base::idx_;
        using base::irrep_;
        using base::dense_irrep_;
        using base::nirrep_;
        using base::layout_;
        using base::factor_;

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

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
            typename dpd_varray_base<U, D, O>::cptr,pointer>>
        explicit indexed_dpd_varray_view(const dpd_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
            typename dpd_varray_base<U, D, O>::pointer,pointer>>
        explicit indexed_dpd_varray_view(dpd_varray_base<U, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
            typename dpd_varray_base<U, D, O>::pointer,pointer>>
        explicit indexed_dpd_varray_view(dpd_varray_base<U, D, O>& other)
        {
            reset(other);
        }
    
        indexed_dpd_varray_view(unsigned irrep, unsigned nirrep,
                                const detail::array_2d<len_type>& len,
                                const detail::array_1d<pointer>& ptr,
                                const detail::array_1d<unsigned>& idx_irrep,
                                const detail::array_2d<len_type>& idx,
                                dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        indexed_dpd_varray_view(unsigned irrep, unsigned nirrep,
                                const detail::array_2d<len_type>& len,
                                const detail::array_1d<pointer>& ptr,
                                const detail::array_1d<unsigned>& idx_irrep,
                                const detail::array_2d<len_type>& idx,
                                const detail::array_1d<unsigned>& depth,
                                layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr, idx_irrep, idx, depth, layout);
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
        using base::factors;
        using base::factor;
        using base::indices;
        using base::index;
        using base::dense_length;
        using base::dense_lengths;
        using base::indexed_length;
        using base::indexed_lengths;
        using base::length;
        using base::lengths;
        using base::indexed_irrep;
        using base::indexed_irreps;
        using base::irrep;
        using base::dense_irrep;
        using base::num_irreps;
        using base::num_indices;
        using base::permutation;
        using base::dimension;
        using base::dense_dimension;
        using base::indexed_dimension;

        Type& factor(len_type idx)
        {
            return const_cast<Type&>(const_cast<const indexed_dpd_varray_view&>(*this).factor(idx));
        }

        void data(const detail::array_1d<pointer>& x)
        {
            MARRAY_ASSERT(x.size() == num_indices());
            x.slurp(data_);
        }

        void data(len_type idx, pointer x)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            data_[idx] = x;
        }

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
