#ifndef MARRAY_DPD_MARRAY_VIEW_HPP
#define MARRAY_DPD_MARRAY_VIEW_HPP

#include "dpd_marray_base.hpp"

MARRAY_BEGIN_NAMESPACE

template <typename Type>
class dpd_marray_view
: public dpd_marray_base<Type, dpd_marray_view<Type>, false>
{
    template <typename, typename, bool> friend class dpd_marray_base;
    template <typename> friend class dpd_marray_view;
    template <typename, typename> friend class dpd_marray;
    template <typename, typename, bool> friend class indexed_dpd_marray_base;

  protected:
    typedef dpd_marray_base<Type, dpd_marray_view, false> base;

    using base::data_;
    using base::irrep_;
    using base::leaf_;
    using base::len_;
    using base::nirrep_;
    using base::off_;
    using base::parent_;
    using base::perm_;
    using base::size_;

    dpd_marray_view(const detail::dpd_base& other,
                    int irrep,
                    typename base::pointer data)
    {
        detail::dpd_base::reset(other);
        irrep_ = irrep;
        data_ = data;
    }

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

    dpd_marray_view() {}

    dpd_marray_view(const dpd_marray_view& other) { reset(other); }

    dpd_marray_view(dpd_marray_view&& other) { reset(std::move(other)); }

    template <typename U, typename D, bool O>
    dpd_marray_view(const dpd_marray_base<U, D, O>& other)
    { reset(other); }

    template <typename U, typename D, bool O>
    dpd_marray_view(dpd_marray_base<U, D, O>& other)
    { reset(other); }

    dpd_marray_view(int irrep,
                    int nirrep,
                    const array_2d<len_type>& len,
                    pointer ptr,
                    dpd_layout layout = DEFAULT_LAYOUT)
    { reset(irrep, nirrep, len, ptr, layout); }

    dpd_marray_view(int irrep,
                    int nirrep,
                    const array_2d<len_type>& len,
                    pointer ptr,
                    const array_1d<int>& depth,
                    layout layout = DEFAULT_LAYOUT)
    { reset(irrep, nirrep, len, ptr, depth, layout); }

    /***********************************************************************
     *
     * Base operations
     *
     **********************************************************************/

    dpd_marray_view& operator=(const dpd_marray_view& other)
    { return base::operator=(other); }

    using base::operator=;
    using base::cview;
    using base::permuted;
    using base::reset;
    using base::view;
    using base::operator();
    using base::cdata;
    using base::data;
    using base::dimension;
    using base::irrep;
    using base::length;
    using base::lengths;
    using base::num_irreps;
    using base::permutation;
    using base::size;

    /***********************************************************************
     *
     * Mutating permute
     *
     **********************************************************************/

    void permute(const array_1d<int>& perm)
    {
        auto ndim = dimension();

        MARRAY_ASSERT(perm.size() == ndim);

        dim_vector new_perm(ndim);
        dim_vector perm_vec;
        perm.slurp(perm_vec);

        for (auto i : range(ndim)) new_perm[i] = perm_[perm_vec[i]];

        perm_ = new_perm;
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

    void swap(dpd_marray_view& other) { base::swap(other); }

    friend void swap(dpd_marray_view& a, dpd_marray_view& b) { a.swap(b); }
};

MARRAY_END_NAMESPACE

#endif // MARRAY_DPD_MARRAY_VIEW_HPP
