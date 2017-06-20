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
        row<pointer> real_data_;
        matrix<len_type> real_idx_;
        struct : Allocator { stride_type size = 0; } storage_;

        template <typename U> using initializer_matrix =
            std::initializer_list<std::initializer_list<U>>;

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

        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           initializer_matrix<len_type> len,
                           std::initializer_list<unsigned> idx_irrep,
                           initializer_matrix<len_type> idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, value, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                detail::is_container_of<V,len_type>::value>>
        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           std::initializer_list<U> len,
                           std::initializer_list<unsigned> idx_irrep,
                           std::initializer_list<V> idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, value, layout);
        }

        template <typename U, typename V, typename W, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value &&
                                (detail::is_container_of_containers_of<W,len_type>::value ||
                                 detail::is_matrix_of<W,len_type>::value)>>
        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           const U& len, const V& idx_irrep, const W& idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, value, layout);
        }

        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           initializer_matrix<len_type> len,
                           std::initializer_list<unsigned> idx_irrep,
                           initializer_matrix<len_type> idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                detail::is_container_of<V,len_type>::value>>
        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           std::initializer_list<U> len,
                           std::initializer_list<unsigned> idx_irrep,
                           std::initializer_list<V> idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        template <typename U, typename V, typename W, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value &&
                                (detail::is_container_of_containers_of<W,len_type>::value ||
                                 detail::is_matrix_of<W,len_type>::value)>>
        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           const U& len, const V& idx_irrep, const W& idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           initializer_matrix<len_type> len,
                           std::initializer_list<unsigned> idx_irrep,
                           initializer_matrix<len_type> idx,
                           uninitialized_t, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                detail::is_container_of<V,len_type>::value>>
        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           std::initializer_list<U> len,
                           std::initializer_list<unsigned> idx_irrep,
                           std::initializer_list<V> idx,
                           uninitialized_t, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
        }

        template <typename U, typename V, typename W, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value &&
                                (detail::is_container_of_containers_of<W,len_type>::value ||
                                 detail::is_matrix_of<W,len_type>::value)>>
        indexed_dpd_varray(unsigned irrep, unsigned nirrep,
                           const U& len, const V& idx_irrep, const W& idx,
                           uninitialized_t, dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
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
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            if (storage_.size > 0)
            {
                pointer data_ = real_data_[0];
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_+i);
                }
                alloc_traits::deallocate(storage_, data_, storage_.size);
                storage_.size = 0;
            }

            real_data_.reset();
            real_idx_.reset();
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
            reset(other, other.layout_);
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

        void reset(unsigned irrep, unsigned nirrep,
                           initializer_matrix<len_type> len,
                           std::initializer_list<unsigned> idx_irrep,
                           initializer_matrix<len_type> idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
            std::uninitialized_fill_n(real_data_[0], storage_.size, value);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                detail::is_container_of<V,len_type>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                           std::initializer_list<U> len,
                           std::initializer_list<unsigned> idx_irrep,
                           std::initializer_list<V> idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
            std::uninitialized_fill_n(real_data_[0], storage_.size, value);
        }

        template <typename U, typename V, typename W, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value &&
                                (detail::is_container_of_containers_of<W,len_type>::value ||
                                 detail::is_matrix_of<W,len_type>::value)>>
         void reset(unsigned irrep, unsigned nirrep,
                           const U& len, const V& idx_irrep, const W& idx,
                           const Type& value = Type(), dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
            std::uninitialized_fill_n(real_data_[0], storage_.size, value);
        }

        void reset(unsigned irrep, unsigned nirrep,
                           initializer_matrix<len_type> len,
                           std::initializer_list<unsigned> idx_irrep,
                           initializer_matrix<len_type> idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                detail::is_container_of<V,len_type>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                           std::initializer_list<U> len,
                           std::initializer_list<unsigned> idx_irrep,
                           std::initializer_list<V> idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        template <typename U, typename V, typename W, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value &&
                                (detail::is_container_of_containers_of<W,len_type>::value ||
                                 detail::is_matrix_of<W,len_type>::value)>>
         void reset(unsigned irrep, unsigned nirrep,
                           const U& len, const V& idx_irrep, const W& idx,
                           dpd_layout layout)
        {
            reset(irrep, nirrep, len, idx_irrep, idx, Type(), layout);
        }

        void reset(unsigned irrep, unsigned nirrep,
                           initializer_matrix<len_type> len,
                           std::initializer_list<unsigned> idx_irrep,
                           initializer_matrix<len_type> idx,
                           uninitialized_t, dpd_layout layout = DEFAULT)
        {
            reset<initializer_matrix<len_type>,
                  std::initializer_list<unsigned>,
                  initializer_matrix<len_type>>(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                detail::is_container_of<V,len_type>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                           std::initializer_list<U> len,
                           std::initializer_list<unsigned> idx_irrep,
                           std::initializer_list<V> idx,
                           uninitialized_t, dpd_layout layout = DEFAULT)
        {
            reset<std::initializer_list<U>,
                  std::initializer_list<unsigned>,
                  std::initializer_list<V>>(irrep, nirrep, len, idx_irrep, idx, uninitialized, layout);
        }

        template <typename U, typename V, typename W, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value &&
                                (detail::is_container_of_containers_of<W,len_type>::value ||
                                 detail::is_matrix_of<W,len_type>::value)>>
        void reset(unsigned irrep, unsigned nirrep,
                   const U& len, const V& idx_irrep, const W& idx,
                   uninitialized_t, dpd_layout layout = DEFAULT)
        {
            len_type idx_ndim = detail::length(idx, 0);
            len_type nidx = detail::length(idx, 1);
            MARRAY_ASSERT((idx_ndim > 0 && nidx > 0) ||
                          (idx_ndim == 0 && nidx == 0));

            real_data_.reset({idx_ndim});
            real_idx_.reset({idx_ndim, nidx}, ROW_MAJOR);
            layout_ = layout;

            detail::set_idx(idx, real_idx_);

            base::reset(irrep, nirrep, len, real_data_.cview(), idx_irrep, real_idx_.cview(), layout);

            stride_type size = dpd_varray_view<Type>::size(dense_irrep_, dense_len_);
            storage_.size = size*idx_ndim;
            real_data_[0] = alloc_traits::allocate(storage_, storage_.size);
            for (len_type i = 1;i < idx_ndim;i++)
                real_data_[i] = real_data_[i-1] + size;
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_dpd_varray& other)
        {
            using std::swap;
            swap(real_data_, other.real_data_);
            swap(real_idx_,  other.real_idx_);
            swap(storage_,   other.storage_);
            base::swap(other);
        }

        friend void swap(indexed_dpd_varray& a, indexed_dpd_varray& b)
        {
            a.swap(b);
        }
};

}

#endif
