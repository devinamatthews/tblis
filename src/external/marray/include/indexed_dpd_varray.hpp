#ifndef _MARRAY_INDEXED_DPD_VARRAY_HPP_
#define _MARRAY_INDEXED_DPD_VARRAY_HPP_

#include "indexed_dpd_varray_view.hpp"

namespace MArray
{

template <typename Type, typename Allocator>
class indexed_dpd_varray : public indexed_dpd_varray_base<Type, indexed_dpd_varray<Type, Allocator>, true>
{
    template <typename, typename, bool> friend class indexed_dpd_varray_base;
    template <typename> friend class indexed_dpd_varray_view;
    template <typename, typename> friend class indexed_dpd_varray;

    protected:
        typedef indexed_dpd_varray_base<Type, indexed_dpd_varray, true> base;
        typedef std::allocator_traits<Allocator> alloc_traits;

    public:
        using typename base::value_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;

    protected:
        using base::size_;
        using base::len_;
        using base::off_;
        using base::stride_;
        using base::idx_irrep_;
        using base::leaf_;
        using base::parent_;
        using base::perm_;
        using base::data_;
        using base::idx_len_;
        using base::idx_;
        using base::irrep_;
        using base::dense_irrep_;
        using base::nirrep_;
        using base::factor_;
        struct : Allocator { stride_type size = 0; } storage_;

    public:

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        indexed_dpd_varray()
        {
            reset();
        }

        indexed_dpd_varray(const indexed_dpd_varray& other)
        {
            reset(other);
        }

        indexed_dpd_varray(indexed_dpd_varray&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        indexed_dpd_varray(const indexed_dpd_varray<U, A>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_assignable_t<reference,U>>
        indexed_dpd_varray(const indexed_dpd_varray_base<U, D, O>& other, dpd_layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_assignable_t<reference,U>>
        indexed_dpd_varray(const indexed_dpd_varray_base<U, D, O>& other,
                           const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset(other, depth, layout);
        }

        indexed_dpd_varray(int irrep, int nirrep,
                           const detail::array_2d<len_type>& len,
                           const detail::array_1d<int>& idx_irrep,
                           const detail::array_2d<len_type>& idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, value, layout);
        }

        indexed_dpd_varray(int irrep, int nirrep,
                           const detail::array_2d<len_type>& len,
                           const detail::array_1d<int>& idx_irrep,
                           const detail::array_2d<len_type>& idx, const Type& value,
                           const detail::array_1d<int>& depth, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, value, depth, layout);
        }

        indexed_dpd_varray(int irrep, int nirrep,
                           const detail::array_2d<len_type>& len,
                           const detail::array_1d<int>& idx_irrep,
                           const detail::array_2d<len_type>& idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        indexed_dpd_varray(int irrep, int nirrep,
                           const detail::array_2d<len_type>& len,
                           const detail::array_1d<int>& idx_irrep,
                           const detail::array_2d<len_type>& idx,
                           const detail::array_1d<int>& depth, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), depth, layout);
        }

        indexed_dpd_varray(int irrep, int nirrep,
                           const detail::array_2d<len_type>& len,
                           const detail::array_1d<int>& idx_irrep,
                           const detail::array_2d<len_type>& idx,
                           uninitialized_t, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
        }

        indexed_dpd_varray(int irrep, int nirrep,
                           const detail::array_2d<len_type>& len,
                           const detail::array_1d<int>& idx_irrep,
                           const detail::array_2d<len_type>& idx, uninitialized_t,
                           const detail::array_1d<int>& depth, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, depth, layout);
        }

        ~indexed_dpd_varray()
        {
            reset();
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        indexed_dpd_varray& operator=(const indexed_dpd_varray& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
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

        stride_type dense_size() const
        {
            return size_[2*dense_dimension()-2][dense_irrep_];
        }

        stride_type size() const
        {
            return dense_size()*num_indices();
        }

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            if (storage_.size > 0)
            {
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_[0]+i);
                }
                alloc_traits::deallocate(storage_, data_[0], storage_.size);
                storage_.size = 0;
            }

            base::reset();
        }

        void reset(indexed_dpd_varray&& other)
        {
            swap(other);
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const indexed_dpd_varray<U, A>& other)
        {
            reset();
            base::reset(const_cast<indexed_dpd_varray<U, A>&>(other));

            storage_.size = size();
            if (storage_.size > 0)
            {
                data_[0] = alloc_traits::allocate(storage_, storage_.size);
                for (auto i : range(1, data_.size()))
                    data_[i] = data_[i-1] + dense_size();
            }

            base::template operator=<>(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const indexed_dpd_varray_base<U, D, O>& other, dpd_layout layout = DEFAULT)
        {
            if (std::is_scalar<Type>::value)
            {
                reset(other.irrep(), other.num_irreps(), other.lengths(),
                      other.indexed_irreps(), other.idx_, uninitialized, layout);
            }
            else
            {
                reset(other.irrep(), other.num_irreps(), other.lengths(),
                      other.indexed_irreps(), other.idx_, layout);
            }

            base::template operator=<>(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const indexed_dpd_varray_base<U, D, O>& other,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            if (std::is_scalar<Type>::value)
            {
                reset(other.irrep(), other.num_irreps(), other.lengths(),
                      other.indexed_irreps(), other.idx_, uninitialized, depth, layout);
            }
            else
            {
                reset(other.irrep(), other.num_irreps(), other.lengths(),
                      other.indexed_irreps(), other.idx_, depth, layout);
            }

            base::template operator=<>(other);
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& idx_irrep,
                   const detail::array_2d<len_type>& idx,
                   const Type& value = Type(), dpd_layout layout = DEFAULT)
       {
           reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
           if (storage_.size > 0)
               std::uninitialized_fill_n(data_[0], storage_.size, value);
       }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& idx_irrep,
                   const detail::array_2d<len_type>& idx, const Type& value,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
       {
           reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, depth, layout);
           if (storage_.size > 0)
               std::uninitialized_fill_n(data_[0], storage_.size, value);
       }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& idx_irrep,
                   const detail::array_2d<len_type>& idx,
                   dpd_layout layout)
       {
           reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
       }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& idx_irrep,
                   const detail::array_2d<len_type>& idx,
                   const detail::array_1d<int>& depth, layout layout)
       {
           reset(irrep, nirrep, len, idx_irrep, idx, Type(), depth, layout);
       }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& idx_irrep,
                   const detail::array_2d<len_type>& idx,
                   uninitialized_t, dpd_layout layout = DEFAULT)
        {
            auto total_ndim = len.length(0);
            auto idx_ndim = idx_irrep.size();
            auto dense_ndim = total_ndim - idx_ndim;

            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized,
                  this->default_depth(layout, dense_ndim), layout.base());
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<int>& idx_irrep,
                   const detail::array_2d<len_type>& idx, uninitialized_t,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
        {
            reset();
            base::reset(irrep, nirrep, len,
                        std::vector<pointer>(idx_irrep.size() ? idx.length(0) : 1),
                        idx_irrep, idx, depth, layout);

            storage_.size = size();
            if (storage_.size > 0)
            {
                data_[0] = alloc_traits::allocate(storage_, storage_.size);
                for (auto i : range(1, data_.size()))
                    data_[i] = data_[i-1] + dense_size();
            }
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_dpd_varray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            base::swap(other);
        }

        friend void swap(indexed_dpd_varray& a, indexed_dpd_varray& b)
        {
            a.swap(b);
        }
};

}

#endif
