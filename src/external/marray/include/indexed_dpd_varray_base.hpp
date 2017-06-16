#ifndef _MARRAY_INDEXED_DPD_VARRAY_BASE_HPP_
#define _MARRAY_INDEXED_DPD_VARRAY_BASE_HPP_

#include "dpd_varray_view.hpp"

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
    template <typename> friend class dpd_varray_view;
    template <typename, typename> friend class dpd_varray;

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
        matrix<len_type> dense_len_;
        matrix<stride_type> dense_size_;
        std::vector<len_type> idx_len_;
        std::vector<unsigned> idx_irrep_;
        std::vector<unsigned> perm_;
        row_view<const pointer> data_;
        matrix_view<const len_type> idx_;
        unsigned irrep_ = 0;
        unsigned dense_irrep_ = 0;
        unsigned nirrep_ = 0;
        dpd_layout layout_ = DEFAULT;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            dense_len_.reset();
            dense_size_.reset();
            idx_len_.clear();
            idx_irrep_.clear();
            perm_.clear();
            data_.reset();
            idx_.reset();
            irrep_ = 0;
            dense_irrep_ = 0;
            nirrep_ = 0;
            layout_ = DEFAULT;
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
            dense_len_.reset(other.dense_len_);
            dense_size_.reset(other.dense_size_);
            idx_len_ = other.idx_len_;
            idx_irrep_ = other.idx_irrep_;
            perm_ = other.perm_;
            data_.reset(other.data_);
            idx_.reset(other.idx_);
            irrep_ = other.irrep_;
            dense_irrep_ = other.dense_irrep_;
            nirrep_ = other.dense_nirrep_;
            layout_ = other.layout_;
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<std::is_assignable<len_type&,U>::value &&
                                std::is_assignable<unsigned&,V>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                   initializer_matrix<U> len, row_view<const pointer> ptr,
                   std::initializer_list<V> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset<initializer_matrix<U>>(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                std::is_assignable<unsigned&,V>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                   std::initializer_list<U> len, row_view<const pointer> ptr,
                   std::initializer_list<V> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset<std::initializer_list<U>>(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<detail::is_container_of_containers_of<U,len_type>::value &&
                                detail::is_container_of<V,unsigned>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                   const U& len, row_view<const pointer> ptr,
                   const V& idx_irrep, matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            unsigned num_idx = ptr.length();
            MARRAY_ASSERT(num_idx > 0);
            MARRAY_ASSERT(idx.length(0) == num_idx);

            unsigned total_ndim = len.size();
            unsigned indexed_ndim = idx_irrep.size();
            unsigned dense_ndim = total_ndim - indexed_ndim;
            MARRAY_ASSERT(total_ndim > indexed_ndim);
            MARRAY_ASSERT(indexed_ndim > 0);
            MARRAY_ASSERT(idx.length(1) == indexed_dim);

            irrep_ = irrep;
            dense_irrep_ = irrep;
            nirrep_ = nirrep;
            data_.reset(ptr);
            idx_.reset(idx);
            layout_ = layout;
            idx_len_.resize(indexed_ndim);
            idx_irrep_.resize(indexed_ndim);
            dense_len_.reset({dense_ndim, nirrep}, ROW_MAJOR);
            dense_size_.reset({num_sizes(dense_ndim, layout), nirrep}, ROW_MAJOR);
            perm_.resize(dense_ndim);

            auto len_it = len.begin();

            if (layout == BLOCKED_COLUMN_MAJOR ||
                layout == PREFIX_COLUMN_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                // Column major
                for (unsigned i = 0;i < dense_ndim;i++)
                {
                    MARRAY_ASSERT(len_it->size() == nirrep);
                    auto it2 = len_it->begin();
                    for (unsigned j = 0;j < nirrep;j++)
                    {
                        dense_len_[i][j] = *it2;
                        ++it2;
                    }
                    perm_[i] = i;
                    ++len_it;
                }
            }
            else
            {
                // Row major: reverse the dimensions and treat as
                // permuted column major
                for (unsigned i = 0;i < dense_ndim;i++)
                {
                    MARRAY_ASSERT(len_it->size() == nirrep);
                    auto it2 = len_it->begin();
                    for (unsigned j = 0;j < nirrep;j++)
                    {
                        dense_len_[dense_ndim-1-i][j] = *it2;
                        ++it2;
                    }
                    perm_[i] = dense_ndim-1-i;
                    ++len_it;
                }
            }

            auto idx_it = idx_irrep.begin();
            for (unsigned i = 0;i < indexed_ndim;i++)
            {
                MARRAY_ASSERT((unsigned)*idx_it < nirrep);
                dense_irrep_ ^= idx_irrep_[i] = *idx_it;
                idx_len_[i] = *std::next(len_it->begin(), idx_irrep_[i]);
                ++idx_it;
                ++len_it;
            }

            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                local_size(dense_irrep_, 0, dense_ndim, 0);
            }
            else
            {
                dense_size_[0] = dense_len_[0];

                for (unsigned i = 1;i < dense_ndim;i++)
                {
                    for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                    {
                        dense_size_[i][irr1] = 0;
                        for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                        {
                            dense_size_[i][irr1] += dense_size_[i-1][irr1^irr2]*
                                                    dense_len_[i][irr2];
                        }
                    }
                }
            }
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<std::is_assignable<len_type&,U>::value &&
                                std::is_assignable<unsigned&,V>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                   matrix_view<U> len, row_view<const pointer> ptr,
                   std::initializer_list<V> idx_irrep,
                   matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            reset<U,std::initializer_list<V>>(irrep, nirrep, len, ptr, idx_irrep, idx, layout);
        }

        template <typename U, typename V, typename=
            detail::enable_if_t<std::is_assignable<len_type&,U>::value &&
                                detail::is_container_of<V,unsigned>::value>>
        void reset(unsigned irrep, unsigned nirrep,
                   matrix_view<U> len, row_view<const pointer> ptr,
                   const V& idx_irrep, matrix_view<const len_type> idx,
                   dpd_layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);
            MARRAY_ASSERT(len.length(1) == nirrep);

            unsigned num_idx = ptr.length();
            MARRAY_ASSERT(num_idx > 0);
            MARRAY_ASSERT(idx.length(0) == num_idx);

            unsigned total_ndim = len.length(0);
            unsigned indexed_ndim = idx_irrep.size();
            unsigned dense_ndim = total_ndim - indexed_ndim;
            MARRAY_ASSERT(total_ndim > indexed_ndim);
            MARRAY_ASSERT(indexed_ndim > 0);
            MARRAY_ASSERT(idx.length(1) == indexed_dim);

            irrep_ = irrep;
            dense_irrep_ = irrep;
            nirrep_ = nirrep;
            data_.reset(ptr);
            idx_.reset(idx);
            layout_ = layout;
            idx_len_.resize(indexed_ndim);
            idx_irrep_.resize(indexed_ndim);
            dense_len_.reset({dense_ndim, nirrep}, ROW_MAJOR);
            dense_size_.reset({num_sizes(dense_ndim, layout), nirrep}, ROW_MAJOR);
            perm_.resize(dense_ndim);

            if (layout == BLOCKED_COLUMN_MAJOR ||
                layout == PREFIX_COLUMN_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                // Column major
                for (unsigned i = 0;i < dense_ndim;i++)
                {
                    for (unsigned j = 0;j < nirrep;j++)
                        dense_len_[i][j] = len[i][j];
                    perm_[i] = i;
                }
            }
            else
            {
                // Row major: reverse the dimensions and treat as
                // permuted column major
                for (unsigned i = 0;i < dense_ndim;i++)
                {
                    for (unsigned j = 0;j < nirrep;j++)
                        dense_len_[dense_ndim-1-i][j] = len[i][j];
                    perm_[i] = dense_ndim-1-i;
                }
            }

            auto idx_it = idx_irrep.begin();
            for (unsigned i = 0;i < indexed_ndim;i++)
            {
                MARRAY_ASSERT((unsigned)*idx_it < nirrep);
                dense_irrep_ ^= idx_irrep_[i] = *idx_it;
                idx_len_[i] = len[dense_ndim+i][idx_irrep_[i]];
                ++idx_it;
            }

            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                local_size(dense_irrep_, 0, dense_ndim, 0);
            }
            else
            {
                dense_size_[0] = dense_len_[0];

                for (unsigned i = 1;i < dense_ndim;i++)
                {
                    for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                    {
                        dense_size_[i][irr1] = 0;
                        for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                        {
                            dense_size_[i][irr1] += dense_size_[i-1][irr1^irr2]*
                                                    dense_len_[i][irr2];
                        }
                    }
                }
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

            const_pointer cptr;
            std::vector<unsigned> indices(ndim);

            for (len_type i = 0;i < num_indices();i++)
            {
                for (unsigned j = 0;j < ndim;j++)
                    indices[j] = idx_[i][j];

                f(View(dense_len_, data_[i], dense_stride_), indices);
            }
        }

        template <typename View, typename Func, unsigned... I>
        void for_each_index(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            constexpr unsigned NDim = sizeof...(I);
            typedef typename View::pointer Ptr;

            MARRAY_ASSERT(NDim == indexed_dimension());

            const_pointer cptr;
            std::array<unsigned, NDim> indices;

            for (len_type i = 0;i < num_indices();i++)
            {
                for (unsigned j = 0;j < NDim;j++)
                    indices[j] = idx_[i][j];

                f(View(dense_len_, data_[i], dense_stride_), indices[I]...);
            }
        }

        template <typename Tp, typename Func>
        void for_each_element(Func&& f) const
        {
            typedef Tp* Ptr;

            unsigned indexed_ndim = indexed_dimension();
            unsigned dense_ndim = dense_dimension();
            unsigned ndim = dense_ndim + indexed_ndim;

            std::vector<unsigned> indices(ndim);

            for (len_type i = 0;i < num_indices();i++)
            {
                for (unsigned j = 0;j < indexed_ndim;j++)
                    indices[dense_ndim+j] = idx_[i][j];

                Ptr ptr = const_cast<Ptr>(data_[i]);
                for (bool done = false;!done;)
                {
                    f(*ptr, indices);

                    for (unsigned j = 0;j < dense_ndim;j++)
                    {
                        ++indices[j];
                        ptr += dense_stride_[j];

                        if (indices[j] == dense_len_[j])
                        {
                            indices[j] = 0;
                            ptr -= dense_stride_[j]*dense_len_[j];
                            if (j == dense_ndim-1) done = true;
                        }
                        else break;
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

            MARRAY_ASSERT(DenseNDim == indexed_dimension());
            MARRAY_ASSERT(IdxNDim = indexed_dimension());

            std::array<unsigned, IdxNDim> indices;

            for (len_type i = 0;i < num_indices();i++)
            {
                for (unsigned j = 0;j < IdxNDim;j++)
                    indices[j] = idx_[i][j];

                miterator<DenseNDim,1> it(dense_len_, dense_stride_);
                Ptr ptr = const_cast<Ptr>(data_[i]);
                while (!it.next(ptr)) f(*ptr, it.position()[I]..., indices[J]...);
            }
        }

        void swap(indexed_dpd_varray_base& other)
        {
            using std::swap;
            swap(len_, other.len_);
            swap(size_, other.size_);
            swap(perm_, other.perm_);
            swap(data_, other.data_);
            swap(irrep_, other.irrep_);
            swap(nirrep_, other.nirrep_);
            swap(layout_, other.layout_);
        }

    public:

        /***********************************************************************
         *
         * Static helper functions
         *
         **********************************************************************/

        static stride_type size(unsigned irrep, initializer_matrix<len_type> len)
        {
            return size<>(irrep, len);
        }

        template <typename U, typename=detail::enable_if_container_of_containers_of_t<U,len_type>>
        static stride_type size(unsigned irrep, const U& len)
        {
            return dpd_marray_base<Type,1,dpd_marray_view<Type,1>,false>::size(irrep, len);
        }

        template <typename U>
        static stride_type size(unsigned irrep, matrix_view<U> len)
        {
            return dpd_marray_base<Type,1,dpd_marray_view<Type,1>,false>::size(irrep, len);
        }

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
            unsigned ndim = dimension();

            MARRAY_ASSERT(ndim == other.dimension());
            MARRAY_ASSERT(nirrep_ == other.nirrep_);
            MARRAY_ASSERT(irrep_ == other.irrep_);

            for (unsigned i = 0;i < ndim;i++)
            {
                MARRAY_ASSERT(lengths(i) == other.lengths(i));
            }

            if (layout_ == other.layout_ && perm_ == other.perm_)
            {
                std::copy_n(other.data(), size(irrep_, len_.view()), data());
            }
            else
            {
                unsigned mask = nirrep_-1;
                unsigned shift = (nirrep_>1) + (nirrep_>2) + (nirrep_>4);

                unsigned nblocks = 1u << (shift*(ndim-1));
                std::vector<unsigned> irreps(ndim);
                for (unsigned block = 0;block < nblocks;block++)
                {
                    unsigned b = block;
                    irreps[0] = irrep_;
                    for (unsigned i = 1;i < ndim;i++)
                    {
                        irreps[0] ^= irreps[i] = b & mask;
                        b >>= shift;
                    }

                    (*this)(irreps) = other(irreps);
                }
            }

            return static_cast<Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            std::fill_n(data(), size(irrep_, len_.view()));
            return static_cast<Derived&>(*this);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        dpd_varray_view<const Type> cview() const
        {
            return const_cast<indexed_dpd_varray_base&>(*this).view();
        }

        dpd_varray_view<ctype> view() const
        {
            return const_cast<indexed_dpd_varray_base&>(*this).view();
        }

        dpd_varray_view<Type> view()
        {
            return *this;
        }

        friend dpd_varray_view<const Type> cview(const indexed_dpd_varray_base& x)
        {
            return x.view();
        }

        friend dpd_varray_view<ctype> view(const indexed_dpd_varray_base& x)
        {
            return x.view();
        }

        friend dpd_varray_view<Type> view(indexed_dpd_varray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        dpd_varray_view<ctype> permuted(std::initializer_list<unsigned> perm) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this).permuted(perm);
        }

        dpd_varray_view<Type> permuted(std::initializer_list<unsigned> perm)
        {
            return permuted<>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        dpd_varray_view<ctype> permuted(const U& perm) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this).permuted<U>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        dpd_varray_view<Type> permuted(const U& perm)
        {
            dpd_varray_view<Type> r(*this);
            r.permute(perm);
            return r;
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        template <typename... Irreps,
            typename=detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value>>
        marray_view<ctype, sizeof...(Irreps)> operator()(const Irreps&... irreps) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this)(irreps...);
        }

        template <typename... Irreps,
            typename=detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value>>
        marray_view<Type, sizeof...(Irreps)> operator()(const Irreps&... irreps_)
        {
            constexpr unsigned NDim = sizeof...(Irreps);

            MARRAY_ASSERT(NDim == dimension());

            std::array<unsigned, NDim> irreps{(unsigned)irreps_...};
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            pointer data;
            get_block(irreps, len, data, stride);

            return marray_view<Type, NDim>(len, data, stride);
        }

        varray_view<ctype> operator()(std::initializer_list<unsigned> irreps) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this)(irreps);
        }

        varray_view<Type> operator()(std::initializer_list<unsigned> irreps)
        {
            return operator()<>(irreps);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<ctype> operator()(const U& irreps) const
        {
            return const_cast<indexed_dpd_varray_base&>(*this)(irreps);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<Type> operator()(const U& irreps_)
        {
            unsigned ndim = dimension();

            std::vector<unsigned> irreps(irreps_.begin(), irreps_.end());
            std::vector<len_type> len(ndim);
            std::vector<stride_type> stride(ndim);

            pointer data;
            get_block(irreps, len, data, stride);

            return varray_view<Type>(len, data, stride);
        }

        template <typename U, typename V, typename W,
            typename=detail::enable_if_t<detail::is_container_of<U,unsigned>::value &&
                                         detail::is_container_of<V,len_type>::value &&
                                         detail::is_container_of<W,stride_type>::value>>
        void get_block(const U& irreps, V& len, const_pointer& data, W& stride) const
        {
            const_cast<indexed_dpd_varray_base&>(*this).get_block(irreps, len,
                                                          const_cast<pointer&>(data),
                                                          stride);
        }

        template <typename U, typename V, typename W,
            typename=detail::enable_if_t<detail::is_container_of<U,unsigned>::value &&
                                         detail::is_container_of<V,len_type>::value &&
                                         detail::is_container_of<W,stride_type>::value>>
        void get_block(const U& irreps, V& len, pointer& data, W& stride)
        {
            unsigned ndim = dimension();

            MARRAY_ASSERT(irreps.size() == ndim);
            MARRAY_ASSERT(len.size() == ndim);
            MARRAY_ASSERT(stride.size() == ndim);

            unsigned irrep = 0;
            for (auto& i : irreps) irrep ^= i;
            MARRAY_ASSERT(irrep == irrep_);

            auto iperm = detail::inverse_permutation(perm_);

            data = data_;

            for (unsigned i = 0;i < ndim;i++)
                len[i] = len_[perm_[i]][irreps[i]];

            if (layout_ == BALANCED_ROW_MAJOR ||
                layout_ == BALANCED_COLUMN_MAJOR)
            {
                local_offset(iperm, irreps, 1, data, stride, 0, ndim, 0);
            }
            else
            {
                if (layout_ == PREFIX_ROW_MAJOR ||
                    layout_ == PREFIX_COLUMN_MAJOR)
                {
                    unsigned lirrep = 0;
                    stride[iperm[0]] = 1;
                    for (unsigned i = 1;i < ndim;i++)
                    {
                        lirrep ^= irreps[iperm[i-1]];
                        stride[iperm[i]] = size_[i-1][lirrep];
                    }

                    unsigned rirrep = irrep_;
                    for (unsigned i = ndim;i --> 1;)
                    {
                        for (unsigned irr1 = 0;irr1 < irreps[iperm[i]];irr1++)
                        {
                            unsigned irr2 = irr1^rirrep;
                            data += size_[i-1][irr2]*len_[i][irr1];
                        }
                        rirrep ^= irreps[iperm[i]];
                    }
                }
                else
                {
                    stride_type lsize = 1;
                    for (unsigned i = 0;i < ndim;i++)
                    {
                        stride[iperm[i]] = lsize;
                        lsize *= len[iperm[i]];
                    }

                    stride_type rsize = 1;
                    unsigned rirrep = irrep_;
                    for (unsigned i = ndim;i --> 1;)
                    {
                        stride_type offset = 0;
                        for (unsigned irr1 = 0;irr1 < irreps[iperm[i]];irr1++)
                        {
                            unsigned irr2 = irr1^rirrep;
                            offset += size_[i-1][irr2]*len_[i][irr1];
                        }
                        data += offset*rsize;
                        rsize *= len[iperm[i]];
                        rirrep ^= irreps[iperm[i]];
                    }
                }
            }
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block<varray_view<ctype>>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block<varray_view<Type>>(std::forward<Func>(f));
        }

        template <unsigned NDim, typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block<marray_view<ctype, NDim>>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <unsigned NDim, typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block<marray_view<Type, NDim>>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_block(
            [&](const varray_view<ctype>& view, const std::vector<unsigned>& irreps)
            {
                view.for_each_element(
                [&](cref value, const std::vector<len_type>& pos)
                {
                    f(value, irreps, pos);
                });
            });
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_block(
            [&](const varray_view<Type>& view, const std::vector<unsigned>& irreps)
            {
                view.for_each_element(
                [&](reference value, const std::vector<len_type>& pos)
                {
                    f(value, irreps, pos);
                });
            });
        }

        template <unsigned NDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<ctype>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <unsigned NDim, typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<Type>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
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
            return const_cast<indexed_varray_base&>(*this).data(idx);
        }

        cptr data(len_type idx) const
        {
            return const_cast<indexed_varray_base&>(*this).data(idx);
        }

        pointer data(len_type idx)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return data_[idx];
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

        len_type dense_length(unsigned dim, unsigned irrep) const
        {
            MARRAY_ASSERT(dim < dense_dimension());
            MARRAY_ASSERT(irrep < nirrep_);
            return dense_len_[perm_[dim]][irrep];
        }

        row_view<const len_type> dense_lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dense_dimension());
            return dense_len_[perm_[dim]];
        }

        matrix<len_type> dense_lengths() const
        {
            unsigned ndim = dense_dimension();
            matrix<len_type> len({ndim, nirrep_}, ROW_MAJOR);
            for (unsigned i = 0;i < ndim;i++) len[i] = dense_len_[perm_[i]];
            return len;
        }

        unsigned indexed_length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < indexed_dimension());
            return idx_len_[dim];
        }

        const std::vector<unsigned>& indexed_lengths() const
        {
            return idx_len_;
        }

        //TODO: length(dim, irrep) = 0 if indexed and wrong irrep,
        //TODO: lengths(dim) and lengths()

        unsigned irrep() const
        {
            return irrep_;
        }

        unsigned num_irreps() const
        {
            return dense_len_.length(1);
        }

        len_type num_indices() const
        {
            return idx_.length(0);
        }

        const std::vector<unsigned>& permutation() const
        {
            return perm_;
        }

        unsigned dimension() const
        {
            return dense_dimension() + indexed_dimension();
        }

        unsigned dense_dimension() const
        {
            return dense_len_.length(0);
        }

        unsigned indexed_dimension() const
        {
            return idx_irrep_.size();
        }
};

}

#endif
