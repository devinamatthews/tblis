#ifndef _MARRAY_INDEXED_DPD_VARRAY_BASE_HPP_
#define _MARRAY_INDEXED_DPD_VARRAY_BASE_HPP_

#include "dpd_marray_view.hpp"
#include "dpd_varray_view.hpp"
#include "indexed_varray_base.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class indexed_dpd_varray_base;

template <typename Type>
class indexed_dpd_varray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class indexed_dpd_varray;

template <typename Type, typename Derived, bool Owner>
class indexed_dpd_varray_base : detail::dpd_base<indexed_dpd_varray_base<Type, Derived, Owner>>
{
    template <typename> friend struct detail::dpd_base;
    template <typename, typename, bool> friend class indexed_dpd_varray_base;
    template <typename> friend class indexed_dpd_varray_view;
    template <typename, typename> friend class indexed_dpd_varray;

    public:
        typedef Type value_type;
        typedef Type* pointer;
        typedef const Type* const_pointer;
        typedef Type& reference;
        typedef const Type& const_reference;

        typedef typename std::conditional<Owner,const Type,Type>::type ctype;
        typedef ctype& cref;
        typedef ctype* cptr;

    protected:
        matrix<stride_type> size_;
        matrix<len_type> len_;
        matrix<len_type> off_;
        matrix<stride_type> stride_;
        irrep_vector idx_irrep_;
        dim_vector leaf_;
        dim_vector parent_;
        dim_vector perm_;
        dim_vector depth_;
        std::vector<pointer> data_;
        matrix<len_type> idx_len_;
        matrix<len_type> idx_;
        unsigned irrep_ = 0;
        unsigned dense_irrep_ = 0;
        unsigned nirrep_ = 0;
        layout layout_ = DEFAULT;
        std::vector<typename std::remove_const<Type>::type> factor_;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            size_.reset();
            len_.reset();
            off_.reset();
            stride_.reset();
            idx_irrep_.clear();
            leaf_.clear();
            parent_.clear();
            perm_.clear();
            depth_.clear();
            data_.clear();
            idx_len_.reset();
            idx_.reset();
            irrep_ = 0;
            dense_irrep_ = 0;
            nirrep_ = 0;
            layout_ = DEFAULT;
            factor_.clear();
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_dpd_varray_base<U, D, O>::cptr,pointer>>
        void reset(const indexed_dpd_varray_base<U, D, O>& other)
        {
            reset(const_cast<indexed_dpd_varray_base<U, D, O>&>(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_dpd_varray_base<U, D, O>::pointer,pointer>>
        void reset(indexed_dpd_varray_base<U, D, O>& other)
        {
            size_.reset(other.size_);
            len_.reset(other.len_);
            off_.reset(other.off_);
            stride_.reset(other.stride_);
            idx_irrep_ = other.idx_irrep_;
            leaf_ = other.leaf_;
            parent_ = other.parent_;
            perm_ = other.perm_;
            depth_ = other.depth_;
            data_.assign(other.data_.begin(), other.data_.end());
            idx_len_.reset(other.idx_len_);
            idx_.reset(other.idx_);
            irrep_ = other.irrep_;
            dense_irrep_ = other.dense_irrep_;
            nirrep_ = other.nirrep_;
            layout_ = other.layout_;
            factor_ = other.factor_;
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::cptr,pointer>>
        void reset(const dpd_varray_base<U, D, O>& other)
        {
            reset(const_cast<dpd_varray_base<U, D, O>&>(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::pointer,pointer>>
        void reset(dpd_varray_base<U, D, O>& other)
        {
            size_.reset(other.size_);
            len_.reset(other.len_);
            off_.reset(other.off_);
            stride_.reset(other.stride_);
            idx_irrep_ = {};
            leaf_ = other.leaf_;
            parent_ = other.parent_;
            perm_ = other.perm_;
            depth_ = other.depth_;
            data_ = {other.data_};
            idx_len_.reset();
            idx_.reset();
            irrep_ = other.irrep_;
            dense_irrep_ = other.irrep_;
            nirrep_ = other.nirrep_;
            layout_ = other.layout_;
            factor_ = {1};
        }

        void reset(unsigned irrep, unsigned nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<pointer>& ptr,
                   const detail::array_1d<unsigned>& idx_irrep,
                   const detail::array_2d<len_type>& idx,
                   dpd_layout layout = DEFAULT)
        {
            unsigned total_ndim = len.length(0);
            unsigned idx_ndim = idx_irrep.size();
            unsigned dense_ndim = total_ndim - idx_ndim;

            reset(irrep, nirrep, len, ptr, idx_irrep, idx,
                  this->default_depth(layout, dense_ndim), layout.base());
        }

        void reset(unsigned irrep, unsigned nirrep,
                   const detail::array_2d<len_type>& len,
                   const detail::array_1d<pointer>& ptr,
                   const detail::array_1d<unsigned>& idx_irrep,
                   const detail::array_2d<len_type>& idx,
                   const detail::array_1d<unsigned>& depth, layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            unsigned total_ndim = len.length(0);
            unsigned idx_ndim = idx_irrep.size();
            unsigned dense_ndim = total_ndim - idx_ndim;
            MARRAY_ASSERT(total_ndim > idx_ndim);
            MARRAY_ASSERT(idx.length(1) == idx_ndim);
            MARRAY_ASSERT(len.length(1) == nirrep);

            unsigned num_idx = ptr.size();
            MARRAY_ASSERT(num_idx > 0);
            MARRAY_ASSERT(idx_ndim > 0 || num_idx == 1);
            MARRAY_ASSERT(idx.length(0) == num_idx || idx_ndim == 0);

            irrep_ = irrep;
            dense_irrep_ = irrep;
            nirrep_ = nirrep;
            ptr.slurp(data_);
            idx.slurp(idx_, ROW_MAJOR);
            layout_ = layout;
            idx_irrep.slurp(idx_irrep_);
            idx_len_.reset({idx_ndim, nirrep}, ROW_MAJOR);
            size_.reset({2*dense_ndim-1, nirrep}, ROW_MAJOR);
            len.slurp(len_, ROW_MAJOR);
            off_.reset({dense_ndim, nirrep}, ROW_MAJOR);
            stride_.reset({dense_ndim, nirrep}, 1, ROW_MAJOR);
            leaf_.resize(dense_ndim);
            parent_.resize(2*dense_ndim-1);
            perm_.resize(dense_ndim);
            depth.slurp(depth_);
            factor_.assign(num_idx, Type(1));

            this->set_tree();
            this->set_size();

            for (unsigned i = 0;i < idx_ndim;i++)
            {
                MARRAY_ASSERT(idx_irrep_[i] < nirrep);
                dense_irrep_ ^= idx_irrep_[i];
                std::copy_n(&len_[i+dense_ndim][0], nirrep, &idx_len_[i][0]);
            }
        }

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename View, typename Func>
        void for_each_index(Func&& f) const
        {
            typedef typename View::pointer Ptr;

            unsigned ndim = indexed_dimension();
            index_vector indices(ndim);
            auto len = dense_lengths();

            for (len_type i = 0;i < num_indices();i++)
            {
                std::copy_n(idx_[i].data(), ndim, indices.data());
                detail::call(std::forward<Func>(f),
                             View(dense_irrep_, nirrep_, len, const_cast<Ptr>(data_[i]), depth_, layout_),
                             indices);
            }
        }

        template <typename View, typename Func, unsigned... I>
        void for_each_index(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            constexpr unsigned NDim = sizeof...(I);
            typedef typename View::pointer Ptr;

            MARRAY_ASSERT(NDim == indexed_dimension());

            auto len = dense_lengths();

            for (len_type i = 0;i < num_indices();i++)
            {
                detail::call(std::forward<Func>(f),
                             View(dense_irrep_, nirrep_, len, const_cast<Ptr>(data_[i]), depth_, layout_),
                             idx_[i][I]...);
            }
        }

        template <typename Tp, typename Func>
        void for_each_element(Func&& f) const
        {
            typedef Tp* Ptr;

            unsigned indexed_ndim = indexed_dimension();
            unsigned dense_ndim = dense_dimension();
            unsigned ndim = dense_ndim + indexed_ndim;

            index_vector indices(ndim);
            irrep_vector irreps(ndim);
            len_vector len(dense_ndim);
            stride_vector stride(dense_ndim);
            const_pointer cptr;

            irrep_vector nirrep(dense_ndim-1, nirrep_);
            viterator<0> it1(nirrep);

            for (unsigned j = 0;j < indexed_ndim;j++)
                irreps[dense_ndim+j] = idx_irrep_[j];

            for (len_type i = 0;i < num_indices();i++)
            {
                for (unsigned j = 0;j < indexed_ndim;j++)
                    indices[dense_ndim+j] = idx_[i][j];

                while (it1.next())
                {
                    irreps[0] = dense_irrep_;
                    for (unsigned j = 1;j < dense_ndim;j++)
                    {
                        irreps[0] ^= irreps[j] = it1.position()[j-1];
                    }

                    cptr = data_[i];
                    this->get_block(irreps, len, cptr, stride);

                    bool skip = false;
                    for (auto l : len) if (l == 0) skip = true;
                    if (skip) continue;

                    Ptr ptr = const_cast<Ptr>(cptr);
                    for (bool done = false;!done;)
                    {
                        detail::call(std::forward<Func>(f), *ptr, irreps, indices);

                        for (unsigned j = 0;j < dense_ndim;j++)
                        {
                            ++indices[j];
                            ptr += stride[j];

                            if (indices[j] == len[j])
                            {
                                indices[j] = 0;
                                ptr -= stride[j]*len[j];
                                if (j == dense_ndim-1) done = true;
                            }
                            else break;
                        }
                    }
                }
            }
        }

        template <typename Tp, typename Func, unsigned... I, unsigned... J>
        void for_each_element(Func&& f, detail::integer_sequence<unsigned, I...>,
                              detail::integer_sequence<unsigned, J...>) const
        {
            constexpr unsigned DenseNDim = sizeof...(I);
            constexpr unsigned IdxNDim = sizeof...(J);
            typedef Tp* Ptr;

            MARRAY_ASSERT(DenseNDim == dense_dimension());
            MARRAY_ASSERT(IdxNDim == indexed_dimension());

            std::array<unsigned, DenseNDim> irreps;
            std::array<len_type, DenseNDim> len;
            std::array<stride_type, DenseNDim> stride;
            const_pointer cptr;

            std::array<unsigned, DenseNDim-1> nirrep;
            nirrep.fill(nirrep_);

            miterator<DenseNDim-1, 0> it1(nirrep);

            for (len_type i = 0;i < num_indices();i++)
            {
                while (it1.next())
                {
                    irreps[0] = dense_irrep_;
                    for (unsigned i = 1;i < DenseNDim;i++)
                    {
                        irreps[0] ^= irreps[i] = it1.position()[i-1];
                    }

                    cptr = data_[i];
                    this->get_block(irreps, len, cptr, stride);

                    miterator<DenseNDim, 1> it2(len, stride);
                    Ptr ptr = const_cast<Ptr>(cptr);
                    while (it2.next(ptr)) detail::call(std::forward<Func>(f), *ptr,
                                                       irreps[I]..., idx_irrep_[J]...,
                                                       it2.position()[I]..., idx_[i][J]...);
                }
            }
        }

        template <typename U, typename D, bool O>
        void copy(const indexed_dpd_varray_base<U, D, O>& other) const
        {
            MARRAY_ASSERT(num_irreps() == other.num_irreps());
            MARRAY_ASSERT(num_indices() == other.num_indices());
            MARRAY_ASSERT(dense_dimension() == other.dense_dimension());
            MARRAY_ASSERT(indexed_irreps() == other.indexed_irreps());
            MARRAY_ASSERT(indexed_lengths() == other.indexed_lengths());

            for (unsigned i = 0;i < dense_dimension();i++)
                MARRAY_ASSERT(dense_lengths(i) == other.dense_lengths(i));

            for (len_type i = 0;i < num_indices();i++)
            {
                MARRAY_ASSERT(indices(i) == other.indices(i));

                const_cast<indexed_dpd_varray_base&>(*this)[i] = other[i];
            }
        }

        void copy(const Type& value) const
        {
            for (len_type i = 0;i < num_indices();i++)
            {
                const_cast<indexed_dpd_varray_base&>(*this)[i] = value;
            }
        }

        void swap(indexed_dpd_varray_base& other)
        {
            using std::swap;
            swap(size_, other.size_);
            swap(len_, other.len_);
            swap(off_, other.off_);
            swap(stride_, other.stride_);
            swap(idx_irrep_, other.idx_irrep_);
            swap(leaf_, other.leaf_);
            swap(parent_, other.parent_);
            swap(perm_, other.perm_);
            swap(depth_, other.depth_);
            swap(data_, other.data_);
            swap(idx_len_, other.idx_len_);
            swap(idx_, other.idx_);
            swap(irrep_, other.irrep_);
            swap(dense_irrep_, other.dense_irrep_);
            swap(nirrep_, other.nirrep_);
            swap(factor_, other.factor_);
            swap(layout_, other.layout_);
            swap(factor_, other.factor_);
        }

    public:

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const indexed_dpd_varray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const indexed_dpd_varray_base<U, D, O>& other)
        {
            copy(other);
            return static_cast<Derived&>(*this);
        }

        template <typename U, typename D, bool O, bool O_=Owner,
            typename=detail::enable_if_t<!O_ && std::is_assignable<reference,U>::value>>
        const Derived& operator=(const indexed_dpd_varray_base<U, D, O>& other) const
        {
            copy(other);
            return static_cast<const Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            copy(value);
            return static_cast<Derived&>(*this);
        }

        template <bool O=Owner, typename=detail::enable_if_t<!O>>
        const Derived& operator=(const Type& value) const
        {
            copy(value);
            return static_cast<const Derived&>(*this);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        indexed_dpd_varray_view<const Type> cview() const
        {
            return *this;
        }

        indexed_dpd_varray_view<ctype> view() const
        {
            return *this;
        }

        indexed_dpd_varray_view<Type> view()
        {
            return *this;
        }

        friend indexed_dpd_varray_view<const Type> cview(const indexed_dpd_varray_base& x)
        {
            return x;
        }

        friend indexed_dpd_varray_view<ctype> view(const indexed_dpd_varray_base& x)
        {
            return x;
        }

        friend indexed_dpd_varray_view<Type> view(indexed_dpd_varray_base& x)
        {
            return x;
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        dpd_varray_view<ctype> operator[](len_type idx) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this)[idx];
        }

        dpd_varray_view<Type> operator[](len_type idx)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return dpd_varray_view<Type>{dense_irrep_, nirrep_, dense_lengths(), data_[idx], depth_, layout_};
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_index(Func&& f) const
        {
            for_each_index<dpd_varray_view<ctype>>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_index(Func&& f)
        {
            for_each_index<dpd_varray_view<Type>>(std::forward<Func>(f));
        }

        template <unsigned DenseNDim, unsigned IdxNDim, typename Func>
        void for_each_index(Func&& f) const
        {
            MARRAY_ASSERT(DenseNDim == dense_dimension());
            for_each_index<dpd_marray_view<ctype, DenseNDim>>(std::forward<Func>(f),
                detail::static_range<unsigned, IdxNDim>{});
        }

        template <unsigned DenseNDim, unsigned IdxNDim, typename Func>
        void for_each_index(Func&& f)
        {
            MARRAY_ASSERT(DenseNDim == dense_dimension());
            for_each_index<dpd_marray_view<Type, DenseNDim>>(std::forward<Func>(f),
                detail::static_range<unsigned, IdxNDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<ctype>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<Type>(std::forward<Func>(f));
        }

        template <unsigned DenseNDim, unsigned IdxNDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<ctype>(std::forward<Func>(f),
                                    detail::static_range<unsigned, DenseNDim>{},
                                    detail::static_range<unsigned, IdxNDim>{});
        }

        template <unsigned DenseNDim, unsigned IdxNDim, typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<Type>(std::forward<Func>(f),
                                   detail::static_range<unsigned, DenseNDim>{},
                                   detail::static_range<unsigned, IdxNDim>{});
        }

        /***********************************************************************
         *
         * Basic getters
         *
         **********************************************************************/

        const std::vector<const_pointer>& cdata() const
        {
            return data();
        }

        const std::vector<const_pointer>& data() const
        {
            return reinterpret_cast<const std::vector<const_pointer>&>(data_);
        }

        const std::vector<pointer>& data()
        {
            return data_;
        }

        const_pointer cdata(len_type idx) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this).data(idx);
        }

        cptr data(len_type idx) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this).data(idx);
        }

        pointer data(len_type idx)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return data_[idx];
        }

        const std::vector<typename std::remove_const<Type>::type>& factors() const
        {
            return factor_;
        }

        const Type& factor(len_type idx) const
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return factor_[idx];
        }

        Type& factor(len_type idx)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return factor_[idx];
        }

        matrix_view<const len_type> indices() const
        {
            return idx_;
        }

        row_view<const len_type> indices(len_type idx) const
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return idx_[idx];
        }

        len_type index(len_type idx, len_type dim) const
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            MARRAY_ASSERT(dim < indexed_dimension());
            return idx_[idx][dim];
        }

        len_type dense_length(unsigned dim, unsigned irrep) const
        {
            MARRAY_ASSERT(dim < dense_dimension());
            MARRAY_ASSERT(irrep < nirrep_);
            return len_[perm_[dim]][irrep];
        }

        row_view<const len_type> dense_lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dense_dimension());
            return len_[perm_[dim]];
        }

        matrix<len_type> dense_lengths() const
        {
            unsigned ndim = dense_dimension();
            matrix<len_type> len({ndim, nirrep_}, ROW_MAJOR);
            for (unsigned i = 0;i < ndim;i++) len[i] = size_[leaf_[perm_[i]]];
            return len;
        }

        len_type indexed_length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < indexed_dimension());
            return idx_len_[dim][idx_irrep_[dim]];
        }

        len_vector indexed_lengths() const
        {
            unsigned idx_ndim = indexed_dimension();
            len_vector len(idx_ndim);
            for (unsigned dim = 0;dim < idx_ndim;dim++)
                len[dim] = idx_len_[dim][idx_irrep_[dim]];
            return len;
        }

        len_type length(unsigned dim, unsigned irrep) const
        {
            MARRAY_ASSERT(dim < dimension());
            MARRAY_ASSERT(irrep < nirrep_);

            if (dim < dense_dimension())
            {
                return dense_length(dim, irrep);
            }
            else
            {
                return idx_len_[dim-dense_dimension()][irrep];
            }

            return 0;
        }

        row_view<const len_type> lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dimension());

            if (dim < dense_dimension())
            {
                return len_[perm_[dim]];
            }
            else
            {
                return idx_len_[dim-dense_dimension()];
            }
        }

        matrix<len_type> lengths() const
        {
            unsigned dense_ndim = dense_dimension();
            unsigned idx_ndim = indexed_dimension();

            matrix<len_type> len({dense_ndim + idx_ndim, nirrep_}, ROW_MAJOR);

            for (unsigned i = 0;i < dense_ndim;i++)
                len[i] = len_[perm_[i]];

            for (unsigned i = 0;i < idx_ndim;i++)
                len[i+dense_ndim] = idx_len_[i];

            return len;
        }

        unsigned indexed_irrep(unsigned dim) const
        {
            MARRAY_ASSERT(dim < indexed_dimension());
            return idx_irrep_[dim];
        }

        const irrep_vector& indexed_irreps() const
        {
            return idx_irrep_;
        }

        unsigned irrep() const
        {
            return irrep_;
        }

        unsigned dense_irrep() const
        {
            return dense_irrep_;
        }

        unsigned num_irreps() const
        {
            return nirrep_;
        }

        len_type num_indices() const
        {
            return std::max<len_type>(1, idx_.length(0));
        }

        const dim_vector& permutation() const
        {
            return perm_;
        }

        unsigned dimension() const
        {
            return dense_dimension() + indexed_dimension();
        }

        unsigned dense_dimension() const
        {
            return perm_.size();
        }

        unsigned indexed_dimension() const
        {
            return idx_irrep_.size();
        }
};

}

#endif
