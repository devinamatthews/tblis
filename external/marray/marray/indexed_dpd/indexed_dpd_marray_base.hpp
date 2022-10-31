#ifndef MARRAY_INDEXED_DPD_MARRAY_BASE_HPP
#define MARRAY_INDEXED_DPD_MARRAY_BASE_HPP

#include "../dpd/dpd_marray.hpp"
#include "../indexed/indexed_marray.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class indexed_dpd_marray_base;

template <typename Type>
class indexed_dpd_marray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class indexed_dpd_marray;

template <typename Type, typename Derived, bool Owner>
class indexed_dpd_marray_base : detail::dpd_base
{
    template <typename, typename, bool> friend class indexed_dpd_marray_base;
    template <typename> friend class indexed_dpd_marray_view;
    template <typename, typename> friend class indexed_dpd_marray;

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
        typedef detail::dpd_base base;
        irrep_vector idx_irrep_;
        std::vector<pointer> data_;
        dpd_len_vector idx_len_;
        matrix<len_type> idx_;
        int dense_irrep_ = 0;
        std::vector<typename std::remove_const<Type>::type> factor_;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            base::reset();
            idx_irrep_.clear();
            data_.clear();
            idx_len_.clear();
            idx_.reset();
            dense_irrep_ = 0;
            factor_.clear();
        }

        template <typename U, bool O, typename D>
        void reset(const indexed_dpd_marray_base<U, D, O>& other)
        {
            reset(const_cast<indexed_dpd_marray_base<U, D, O>&>(other));
        }

        template <typename U, bool O, typename D>
        void reset(indexed_dpd_marray_base<U, D, O>& other)
        {
            base::reset(other);
            idx_irrep_ = other.idx_irrep_;
            data_.assign(other.data_.begin(), other.data_.end());
            idx_len_ = other.idx_len_;
            idx_.reset(other.idx_);
            dense_irrep_ = other.dense_irrep_;
            factor_ = other.factor_;
        }

        template <typename U, bool O, typename D>
        void reset(const dpd_marray_base<U, D, O>& other)
        {
            reset(const_cast<dpd_marray_base<U, D, O>&>(other));
        }

        template <typename U, bool O, typename D>
        void reset(dpd_marray_base<U, D, O>& other)
        {
            base::reset(other);
            idx_irrep_ = {};
            data_ = {other.data_};
            idx_len_.clear();
            idx_.reset();
            dense_irrep_ = other.irrep_;
            factor_.assign(1, Type(1));
        }

        void reset(int irrep, int nirrep,
                   const array_2d<len_type>& len,
                   const array_1d<pointer>& ptr,
                   const array_1d<int>& idx_irrep,
                   const array_2d<len_type>& idx,
                   dpd_layout layout = DEFAULT_LAYOUT)
        {
            auto total_ndim = len.length(0);
            auto idx_ndim = idx_irrep.size();
            auto dense_ndim = total_ndim - idx_ndim;

            reset(irrep, nirrep, len, ptr, idx_irrep, idx,
                  default_depth(layout, dense_ndim), layout.base());
        }

        void reset(int irrep, int nirrep,
                   const array_2d<len_type>& len,
                   const array_1d<pointer>& ptr,
                   const array_1d<int>& idx_irrep,
                   const array_2d<len_type>& idx,
                   const array_1d<int>& depth, layout layout = DEFAULT_LAYOUT)
        {
            base::reset(irrep, nirrep, len, depth, layout);

            auto total_ndim = len.length(0);
            auto idx_ndim = idx_irrep.size();
            auto dense_ndim = total_ndim - idx_ndim;
            MARRAY_ASSERT(total_ndim > idx_ndim);
            MARRAY_ASSERT(idx.length(1) == idx_ndim);
            MARRAY_ASSERT(len.length(1) >= nirrep);
            MARRAY_ASSERT(depth.size() == dense_ndim);

            auto num_idx = ptr.size();
            MARRAY_ASSERT(num_idx > 0);
            MARRAY_ASSERT(idx_ndim > 0 || num_idx == 1);
            MARRAY_ASSERT(idx.length(0) == num_idx || idx_ndim == 0);

            dense_irrep_ = irrep_;
            ptr.slurp(data_);
            idx.slurp(idx_, ROW_MAJOR);
            idx_irrep.slurp(idx_irrep_);
            idx_len_.resize(idx_ndim);
            factor_.assign(num_indices(), Type(1));

            for (auto i : range(idx_ndim))
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
        void for_each_index_(Func&& f) const
        {
            typedef typename View::pointer Ptr;

            auto ndim = indexed_dimension();
            index_vector indices(ndim);
            auto len = dense_lengths();

            for (auto i : range(num_indices()))
            {
                std::copy_n(idx_[i].data(), ndim, indices.data());
                detail::call(std::forward<Func>(f),
                             View(*this, dense_irrep_, const_cast<Ptr>(data_[i])),
                             indices);
            }
        }

        template <typename View, typename Func, int... I>
        void for_each_index_(Func&& f, std::integer_sequence<int, I...>) const
        {
            typedef typename View::pointer Ptr;

            MARRAY_ASSERT(sizeof...(I) == indexed_dimension());

            auto len = dense_lengths();

            for (auto i : range(num_indices()))
            {
                detail::call(std::forward<Func>(f),
                             View(*this, dense_irrep_, const_cast<Ptr>(data_[i])),
                             idx_[i][I]...);
            }
        }

        template <typename Tp, typename Func>
        void for_each_element_(Func&& f) const
        {
            typedef Tp* Ptr;

            auto indexed_ndim = indexed_dimension();
            auto dense_ndim = dense_dimension();
            auto ndim = dense_ndim + indexed_ndim;

            index_vector indices(ndim);
            irrep_vector irreps(ndim);
            len_vector len(dense_ndim);
            stride_vector stride(dense_ndim);
            const_pointer cptr;

            for (auto j : range(indexed_ndim))
                irreps[dense_ndim+j] = idx_irrep_[j];

            for (auto i : range(num_indices()))
            {
                for (auto j : range(indexed_ndim))
                    indices[dense_ndim+j] = idx_[i][j];

                irrep_iterator it1(dense_irrep(), num_irreps(), dense_ndim);
                while (it1.next())
                {
                    for (auto j : range(dense_ndim))
                        irreps[j] = it1.irrep(j);

                    cptr = data_[i];
                    get_block(irreps, len, cptr, stride);

                    bool skip = false;
                    for (auto l : len) if (l == 0) skip = true;
                    if (skip) continue;

                    auto it2 = make_iterator(len, stride);
                    Ptr ptr = const_cast<Ptr>(cptr);
                    while (it2.next(ptr))
                    {
                        for (auto j : range(dense_ndim))
                            indices[j] = it2.position()[j];

                        detail::call(std::forward<Func>(f), *ptr, irreps, indices);
                    }
                }
            }
        }

        template <typename Tp, typename Func, int... I, int... J>
        void for_each_element_(Func&& f, std::integer_sequence<int, I...>,
                               std::integer_sequence<int, J...>) const
        {
            constexpr int DenseNDim = sizeof...(I);
            typedef Tp* Ptr;

            MARRAY_ASSERT(DenseNDim == dense_dimension());
            MARRAY_ASSERT(sizeof...(J) == indexed_dimension());

            std::array<int, DenseNDim> irreps;
            std::array<len_type, DenseNDim> len;
            std::array<stride_type, DenseNDim> stride;
            const_pointer cptr;

            for (len_type i = 0;i < num_indices();i++)
            {
                irrep_iterator it1(dense_irrep(), num_irreps(), DenseNDim);
                while (it1.next())
                {
                    for (auto j : range(DenseNDim))
                        irreps[j] = it1.irrep(j);

                    cptr = data_[i];
                    get_block(irreps, len, cptr, stride);

                    auto it2 = make_iterator(len, stride);
                    Ptr ptr = const_cast<Ptr>(cptr);
                    while (it2.next(ptr))
                        detail::call(std::forward<Func>(f), *ptr,
                                                       irreps[I]..., idx_irrep_[J]...,
                                                       it2.position()[I]..., idx_[i][J]...);
                }
            }
        }

        template <typename U, typename D, bool O>
        void copy_(const indexed_dpd_marray_base<U, D, O>& other) const
        {
            MARRAY_ASSERT(num_irreps() == other.num_irreps());
            MARRAY_ASSERT(num_indices() == other.num_indices());
            MARRAY_ASSERT(dense_dimension() == other.dense_dimension());
            MARRAY_ASSERT(indexed_irreps() == other.indexed_irreps());
            MARRAY_ASSERT(indexed_lengths() == other.indexed_lengths());

            for (auto i : range(dense_dimension()))
                MARRAY_ASSERT(dense_lengths(i) == other.dense_lengths(i));

            for (auto i : range(num_indices()))
            {
                MARRAY_ASSERT(indices(i) == other.indices(i));
                const_cast<indexed_dpd_marray_base&>(*this)[i] = other[i];
            }
        }

        void copy_(const Type& value) const
        {
            for (auto i : range(num_indices()))
                const_cast<indexed_dpd_marray_base&>(*this)[i] = value;
        }

        void swap(indexed_dpd_marray_base& other)
        {
            using std::swap;
            base::swap(other);
            swap(idx_irrep_, other.idx_irrep_);
            swap(data_, other.data_);
            swap(idx_len_, other.idx_len_);
            swap(idx_, other.idx_);
            swap(dense_irrep_, other.dense_irrep_);
            swap(factor_, other.factor_);
        }

    public:

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const indexed_dpd_marray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O,
            typename=std::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const indexed_dpd_marray_base<U, D, O>& other)
        {
            copy_(other);
            return static_cast<Derived&>(*this);
        }

        template <typename U, typename D, bool O, bool O_=Owner,
            typename=std::enable_if_t<!O_ && std::is_assignable<reference,U>::value>>
        const Derived& operator=(const indexed_dpd_marray_base<U, D, O>& other) const
        {
            copy_(other);
            return static_cast<const Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            copy_(value);
            return static_cast<Derived&>(*this);
        }

        template <bool O=Owner, typename=std::enable_if_t<!O>>
        const Derived& operator=(const Type& value) const
        {
            copy_(value);
            return static_cast<const Derived&>(*this);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        indexed_dpd_marray_view<const Type> cview() const
        {
            return *this;
        }

        indexed_dpd_marray_view<ctype> view() const
        {
            return *this;
        }

        indexed_dpd_marray_view<Type> view()
        {
            return *this;
        }

        friend indexed_dpd_marray_view<const Type> cview(const indexed_dpd_marray_base& x)
        {
            return x;
        }

        friend indexed_dpd_marray_view<ctype> view(const indexed_dpd_marray_base& x)
        {
            return x;
        }

        friend indexed_dpd_marray_view<Type> view(indexed_dpd_marray_base& x)
        {
            return x;
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        dpd_marray_view<ctype> operator[](len_type idx) const
        {
            return const_cast<indexed_dpd_marray_base&>(*this)[idx];
        }

        dpd_marray_view<Type> operator[](len_type idx)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return {*this, dense_irrep_, data_[idx]};
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_index(Func&& f) const
        {
            for_each_index_<dpd_marray_view<ctype>>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_index(Func&& f)
        {
            for_each_index_<dpd_marray_view<Type>>(std::forward<Func>(f));
        }

        template <int DenseNDim, int IdxNDim, typename Func>
        void for_each_index(Func&& f) const
        {
            MARRAY_ASSERT(DenseNDim == dense_dimension());
            for_each_index_<dpd_marray_view<ctype>>(std::forward<Func>(f),
                std::make_integer_sequence<int, IdxNDim>{});
        }

        template <int DenseNDim, int IdxNDim, typename Func>
        void for_each_index(Func&& f)
        {
            MARRAY_ASSERT(DenseNDim == dense_dimension());
            for_each_index_<dpd_marray_view<Type>>(std::forward<Func>(f),
                std::make_integer_sequence<int, IdxNDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element_<ctype>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element_<Type>(std::forward<Func>(f));
        }

        template <int DenseNDim, int IdxNDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element_<ctype>(std::forward<Func>(f),
                                     std::make_integer_sequence<int, DenseNDim>{},
                                     std::make_integer_sequence<int, IdxNDim>{});
        }

        template <int DenseNDim, int IdxNDim, typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element_<Type>(std::forward<Func>(f),
                                    std::make_integer_sequence<int, DenseNDim>{},
                                    std::make_integer_sequence<int, IdxNDim>{});
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
            return const_cast<indexed_dpd_marray_base&>(*this).data(idx);
        }

        cptr data(len_type idx) const
        {
            return const_cast<indexed_dpd_marray_base&>(*this).data(idx);
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

        row_view<Type> factors()
        {
            return {{factor_.size()}, factor_.data()};
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

        len_type dense_length(int dim, int irrep) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dense_dimension());
            MARRAY_ASSERT(irrep >= 0 && irrep < nirrep_);
            return len_[perm_[dim]][irrep];
        }

        row_view<const len_type> dense_lengths(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dense_dimension());
            return row_view<const len_type>{{nirrep_}, &len_[perm_[dim]][0]};
        }

        matrix<len_type> dense_lengths() const
        {
            auto ndim = dense_dimension();
            matrix<len_type> len({ndim, num_irreps()}, ROW_MAJOR);
            for (auto i : range(ndim))
                std::copy_n(dense_lengths(i).data(), num_irreps(), &len[i][0]);
            return len;
        }

        len_type indexed_length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < indexed_dimension());
            return idx_len_[dim][idx_irrep_[dim]];
        }

        len_vector indexed_lengths() const
        {
            auto ndim = indexed_dimension();
            len_vector len(ndim);
            for (auto dim : range(ndim))
                len[dim] = idx_len_[dim][idx_irrep_[dim]];
            return len;
        }

        len_type length(int dim, int irrep) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(irrep >= 0 && irrep < nirrep_);

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

        row_view<const len_type> lengths(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());

            if (dim < dense_dimension())
            {
                return dense_lengths(dim);
            }
            else
            {
                return {{nirrep_}, &idx_len_[dim-dense_dimension()][0]};
            }
        }

        matrix<len_type> lengths() const
        {
            auto dense_ndim = dense_dimension();
            auto idx_ndim = indexed_dimension();

            matrix<len_type> len({dense_ndim + idx_ndim, num_irreps()}, ROW_MAJOR);

            for (auto i : range(dense_ndim))
                std::copy_n(dense_lengths(i).data(), num_irreps(), &len[i][0]);

            for (auto i : range(idx_ndim))
                std::copy_n(idx_len_[i].data(), num_irreps(), &len[i+dense_ndim][0]);

            return len;
        }

        auto indexed_irrep(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < indexed_dimension());
            return idx_irrep_[dim];
        }

        auto& indexed_irreps() const
        {
            return idx_irrep_;
        }

        auto irrep() const
        {
            return irrep_;
        }

        auto dense_irrep() const
        {
            return dense_irrep_;
        }

        auto num_irreps() const
        {
            return nirrep_;
        }

        auto num_indices() const
        {
            return std::max<len_type>(1, idx_.length(0));
        }

        auto& permutation() const
        {
            return perm_;
        }

        auto dimension() const
        {
            return dense_dimension() + indexed_dimension();
        }

        int dense_dimension() const
        {
            return perm_.size();
        }

        int indexed_dimension() const
        {
            return idx_irrep_.size();
        }
};

}

#endif //MARRAY_INDEXED_DPD_MARRAY_BASE_HPP
