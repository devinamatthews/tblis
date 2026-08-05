#ifndef MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP
#define MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP

#include "indexed_dpd_marray_base.hpp"

MARRAY_BEGIN_NAMESPACE

template <typename Type>
class indexed_dpd_marray_view
: public indexed_dpd_marray_base<Type, indexed_dpd_marray_view<Type>, false>
{
    template <typename, typename, bool> friend class indexed_dpd_marray_base;
    template <typename> friend class indexed_dpd_marray_view;
    template <typename, typename> friend class indexed_dpd_marray;

  protected:
    typedef indexed_dpd_marray_base<Type, indexed_dpd_marray_view, false> base;

    using base::data_;
    using base::dense_irrep_;
    using base::factor_;
    using base::idx_;
    using base::idx_irrep_;
    using base::idx_len_;
    using base::irrep_;
    using base::leaf_;
    using base::len_;
    using base::nirrep_;
    using base::off_;
    using base::parent_;
    using base::perm_;
    using base::size_;
    using base::stride_;

  public:
    using typename base::const_pointer;
    using typename base::const_reference;
    using typename base::pointer;
    using typename base::reference;
    using typename base::value_type;

    /***********************************************************************
     *
     * Constructors
     *
     **********************************************************************/

    indexed_dpd_marray_view() { reset(); }

    indexed_dpd_marray_view(const indexed_dpd_marray_view& other)
    {
        reset(other);
    }

    indexed_dpd_marray_view(indexed_dpd_marray_view&& other)
    {
        reset(std::move(other));
    }

    template <typename U, bool O, typename D>
    indexed_dpd_marray_view(const indexed_dpd_marray_base<U, D, O>& other)
    {
        reset(other);
    }

    template <typename U, bool O, typename D>
    indexed_dpd_marray_view(indexed_dpd_marray_base<U, D, O>&& other)
    {
        reset(other);
    }

    template <typename U, bool O, typename D>
    indexed_dpd_marray_view(indexed_dpd_marray_base<U, D, O>& other)
    {
        reset(other);
    }

    template <typename U, bool O, typename D>
    explicit indexed_dpd_marray_view(const dpd_marray_base<U, D, O>& other)
    {
        reset(other);
    }

    template <typename U, bool O, typename D>
    explicit indexed_dpd_marray_view(dpd_marray_base<U, D, O>&& other)
    {
        reset(other);
    }

    template <typename U, bool O, typename D>
    explicit indexed_dpd_marray_view(dpd_marray_base<U, D, O>& other)
    {
        reset(other);
    }

    indexed_dpd_marray_view(int irrep,
                            int nirrep,
                            const array_2d<len_type>& len,
                            const array_1d<pointer>& ptr,
                            const array_1d<int>& idx_irrep,
                            const array_2d<len_type>& idx,
                            dpd_layout layout = DEFAULT_LAYOUT)
    {
        reset(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
    }

    indexed_dpd_marray_view(int irrep,
                            int nirrep,
                            const array_2d<len_type>& len,
                            const array_1d<pointer>& ptr,
                            const array_1d<int>& idx_irrep,
                            const array_2d<len_type>& idx,
                            const array_1d<int>& depth,
                            layout layout = DEFAULT_LAYOUT)
    {
        reset(irrep, nirrep, len, ptr, idx_irrep, idx, depth, layout);
    }

    /***********************************************************************
     *
     * Base operations
     *
     **********************************************************************/

    indexed_dpd_marray_view& operator=(const indexed_dpd_marray_view& other)
    {
        return base::operator=(other);
    }

    using base::operator=;
    using base::cview;
    using base::reset;
    using base::view;
    using base::operator[];
    using base::cdata;
    using base::data;
    using base::dense_dimension;
    using base::dense_irrep;
    using base::dense_length;
    using base::dense_lengths;
    using base::dimension;
    using base::factor;
    using base::factors;
    using base::index;
    using base::indexed_dimension;
    using base::indexed_irrep;
    using base::indexed_irreps;
    using base::indexed_length;
    using base::indexed_lengths;
    using base::indices;
    using base::irrep;
    using base::length;
    using base::lengths;
    using base::num_indices;
    using base::num_irreps;
    using base::permutation;

    Type& factor(len_type idx)
    {
        return const_cast<Type&>(
            const_cast<const indexed_dpd_marray_view&>(*this).factor(idx));
    }

    void data(const array_1d<pointer>& x)
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

    void swap(indexed_dpd_marray_view& other) { base::swap(other); }

    friend void swap(indexed_dpd_marray_view& a, indexed_dpd_marray_view& b)
    {
        a.swap(b);
    }
};

MARRAY_END_NAMESPACE

#endif // MARRAY_INDEXED_DPD_MARRAY_VIEW_HPP
