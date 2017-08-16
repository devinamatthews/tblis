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
class indexed_dpd_varray_base
{
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
        template <typename U> using initializer_matrix =
            std::initializer_list<std::initializer_list<U>>;

    protected:
        matrix<len_type> len_;
        matrix<stride_type> dense_size_;
        irrep_vector idx_irrep_;
        dim_vector perm_;
        row_view<const pointer> data_;
        matrix_view<const len_type> idx_;
        unsigned irrep_ = 0;
        unsigned dense_irrep_ = 0;
        unsigned nirrep_ = 0;
        dpd_layout layout_ = DEFAULT;
        std::vector<typename std::remove_const<Type>::type> factor_;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            len_.reset();
            dense_size_.reset();
            idx_irrep_.clear();
            perm_.clear();
            data_.reset();
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
            len_.reset(other.len_);
            dense_size_.reset(other.dense_size_);
            idx_irrep_ = other.idx_irrep_;
            perm_ = other.perm_;
            data_.reset(other.data_);
            idx_.reset(other.idx_);
            irrep_ = other.irrep_;
            dense_irrep_ = other.dense_irrep_;
            nirrep_ = other.nirrep_;
            layout_ = other.layout_;
            factor_ = other.factor_;
        }

        void reset(unsigned irrep, unsigned nirrep,
                   initializer_matrix<len_type> len, row_view<const pointer> ptr,
                   std::initializer_list<unsigned> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset<initializer_matrix<len_type>,
                std::initializer_list<unsigned>>(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename=
            detail::enable_if_container_of_t<U,len_type>>
        void reset(unsigned irrep, unsigned nirrep,
                   std::initializer_list<U> len, row_view<const pointer> ptr,
                   std::initializer_list<unsigned> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset<std::initializer_list<U>,
                  std::initializer_list<unsigned>>(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<(detail::is_container_of_containers_of<U,len_type>::value ||
                                 detail::is_matrix_of<U,len_type>::value) &&
                                detail::is_container_of<V,unsigned>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                   const U& len, row_view<const pointer> ptr,
                   const V& idx_irrep, matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            unsigned num_idx = ptr.length();
            MARRAY_ASSERT(idx.length(0) == num_idx);

            unsigned total_ndim = detail::length(len, 0);
            unsigned idx_ndim = idx_irrep.size();
            unsigned dense_ndim = total_ndim - idx_ndim;
            MARRAY_ASSERT(total_ndim > idx_ndim);
            MARRAY_ASSERT(idx_ndim > 0 || num_idx == 1);
            MARRAY_ASSERT(idx.length(1) == idx_ndim);
            MARRAY_ASSERT(detail::length(len, 1) == nirrep);

            irrep_ = irrep;
            dense_irrep_ = irrep;
            nirrep_ = nirrep;
            data_.reset(ptr);
            idx_.reset(idx);
            layout_ = layout;
            idx_irrep_.resize(idx_ndim);
            len_.reset({total_ndim, nirrep}, ROW_MAJOR);
            dense_size_.reset({2*dense_ndim, nirrep}, ROW_MAJOR);
            perm_.resize(dense_ndim);
            factor_.assign(num_idx, Type(1));

            detail::set_len(len, len_, perm_, layout_);
            detail::set_size(irrep_, len_, dense_size_, layout_);

            unsigned i = 0;
            for (unsigned irrep : idx_irrep)
            {
                MARRAY_ASSERT(irrep < nirrep);
                dense_irrep_ ^= idx_irrep_[i++] = irrep;
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
                             View(dense_irrep_, nirrep_, len, const_cast<Ptr>(data_[i]), layout_),
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
                             View(dense_irrep_, nirrep_, len, const_cast<Ptr>(data_[i]), layout_),
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

            auto iperm = detail::inverse_permutation(perm_);
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
                    for (unsigned i = 1;i < dense_ndim;i++)
                    {
                        irreps[0] ^= irreps[i] = it1.position()[i-1];
                    }

                    cptr = data_[i];
                    detail::get_block(iperm, irreps, len_, dense_size_,
                                      layout_, len, cptr, stride);

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

            auto iperm = detail::inverse_permutation(perm_);
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
                    detail::get_block(iperm, irreps, len_, dense_size_,
                                      layout_, len, cptr, stride);

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

            stride_type size = dpd_varray_view<Type>::size(dense_irrep_, dense_lengths());

            if (layout_ == other.layout_)
            {
                for (len_type i = 0;i < num_indices();i++)
                {
                    MARRAY_ASSERT(indices(i) == other.indices(i));

                    pointer a = const_cast<pointer>(data(i));
                    auto b = other.data(i);

                    std::copy_n(b, size, a);
                }
            }
            else
            {
                for (len_type i = 0;i < num_indices();i++)
                {
                    MARRAY_ASSERT(indices(i) == other.indices(i));

                    const_cast<indexed_dpd_varray_base&>(*this)[i] = other[i];
                }
            }
        }

        void copy(const Type& value) const
        {
            stride_type size = dpd_varray_view<Type>::size(dense_irrep_, dense_lengths());
            std::fill_n(const_cast<pointer>(data(0)), size*num_indices(), value);
        }

        void swap(indexed_dpd_varray_base& other)
        {
            using std::swap;
            swap(len_, other.len_);
            swap(dense_size_, other.dense_size_);
            swap(idx_irrep_, other.idx_irrep_);
            swap(perm_, other.perm_);
            swap(data_, other.data_);
            swap(idx_, other.idx_);
            swap(irrep_, other.irrep_);
            swap(dense_irrep_, other.dense_irrep_);
            swap(nirrep_, other.nirrep_);
            swap(layout_, other.layout_);
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
            return {dense_irrep_, nirrep_, dense_lengths(), data_[idx], layout_};
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

        const row_view<const const_pointer>& cdata() const
        {
            return reinterpret_cast<const row_view<const const_pointer>&>(data_);
        }

        const row_view<const cptr>& data() const
        {
            return reinterpret_cast<const row_view<const cptr>&>(data_);
        }

        const row_view<const pointer>& data()
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

        const std::vector<Type>& factors() const
        {
            return factor_;
        }

        const Type& factor(len_type idx) const
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return factor_[idx];
        }

        const matrix_view<const len_type>& indices() const
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
            for (unsigned i = 0;i < ndim;i++) len[i] = len_[perm_[i]];
            return len;
        }

        len_type indexed_length(unsigned dim) const
        {
            unsigned dense_ndim = dense_dimension();
            unsigned idx_ndim = indexed_dimension();
            MARRAY_ASSERT(dim < idx_ndim);
            return len_[dense_ndim+dim][idx_irrep_[dim]];
        }

        len_vector indexed_lengths() const
        {
            unsigned dense_ndim = dense_dimension();
            unsigned idx_ndim = indexed_dimension();
            len_vector len(idx_ndim);
            for (unsigned dim = 0;dim < idx_ndim;dim++)
                len[dim] = len_[dense_ndim+dim][idx_irrep_[dim]];
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
                return len_[dim][irrep];
            }

            return 0;
        }

        len_vector lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dimension());

            len_vector len(nirrep_);

            if (dim < dense_dimension())
            {
                std::copy_n(&len_[perm_[dim]][0], nirrep_, len.data());
            }
            else
            {
                std::copy_n(&len_[dim][0], nirrep_, len.data());
            }

            return len;
        }

        matrix<len_type> lengths() const
        {
            unsigned dense_ndim = dense_dimension();
            unsigned idx_ndim = indexed_dimension();

            matrix<len_type> len({dense_ndim + idx_ndim, nirrep_}, ROW_MAJOR);

            for (unsigned i = 0;i < dense_ndim;i++)
                len[i] = len_[perm_[i]];

            for (unsigned i = dense_ndim;i < dense_ndim+idx_ndim;i++)
                len[i] = len_[i];

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
            return len_.length(1);
        }

        len_type num_indices() const
        {
            return idx_.length(0);
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
