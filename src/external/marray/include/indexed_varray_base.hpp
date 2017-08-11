#ifndef _MARRAY_INDEXED_VARRAY_BASE_HPP_
#define _MARRAY_INDEXED_VARRAY_BASE_HPP_

#include "varray_view.hpp"
#include "dpd_marray_base.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class indexed_varray_base;

template <typename Type>
class indexed_varray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class indexed_varray;

namespace detail
{

template <typename U, typename V>
enable_if_container_of_containers_of_t<U,len_type>
set_idx(const U& idx_in, V& idx)
{
    unsigned ndim = idx_in.size();
    unsigned nidx = idx_in.begin()->size();

    if (nidx == 0) return;

    auto it = idx_in.begin();

    for (unsigned i = 0;i < ndim;i++)
    {
        std::copy_n(it->begin(), nidx, &idx[i][0]);
        ++it;
    }
}

template <typename U, typename V>
enable_if_matrix_of_t<U,len_type>
set_idx(const U& idx_in, V& idx)
{
    unsigned ndim = idx_in.length(0);
    unsigned nidx = idx_in.length(1);

    if (nidx == 0) return;

    for (unsigned i = 0;i < ndim;i++)
    {
        std::copy_n(&idx_in[i][0], nidx, &idx[i][0]);
    }
}

}

template <typename Type, typename Derived, bool Owner>
class indexed_varray_base
{
    template <typename, typename, bool> friend class indexed_varray_base;
    template <typename> friend class indexed_varray_view;
    template <typename, typename> friend class indexed_varray;

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
        row_view<const pointer> data_;
        matrix_view<const len_type> idx_;
        len_vector dense_len_;
        len_vector idx_len_;
        stride_vector dense_stride_;
        std::vector<Type> factor_;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            data_.reset();
            idx_.reset();
            dense_len_.clear();
            idx_len_.clear();
            dense_stride_.clear();
            factor_.clear();
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_varray_base<U, D, O>::cptr,pointer>>
        void reset(const indexed_varray_base<U, D, O>& other)
        {
            reset(const_cast<indexed_varray_base<U, D, O>&>(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_varray_base<U, D, O>::pointer,pointer>>
        void reset(indexed_varray_base<U, D, O>& other)
        {
            data_.reset(other.data_);
            idx_.reset(other.idx_);
            dense_len_ = other.dense_len_;
            idx_len_ = other.idx_len_;
            dense_stride_ = other.dense_stride_;
            factor_ = other.factor_;
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        void reset(const U& len, row_view<const pointer> ptr,
                   matrix_view<const len_type> idx, layout layout = DEFAULT)
        {
            unsigned num_idx = ptr.length();
            MARRAY_ASSERT(num_idx > 0);
            MARRAY_ASSERT(idx.length(0) == num_idx);

            unsigned total_dim = len.size();
            unsigned idx_dim = idx.length(1);
            unsigned dense_dim = total_dim - idx_dim;
            MARRAY_ASSERT(total_dim > idx_dim);
            MARRAY_ASSERT(idx_dim > 0 || num_idx == 1);

            data_.reset(ptr);
            idx_.reset(idx);
            dense_len_.assign(len.begin(), std::next(len.begin(),dense_dim));
            idx_len_.assign(std::next(len.begin(),dense_dim), len.end());
            dense_stride_ = varray_view<Type>::strides(dense_len_, layout);
            factor_.assign(num_idx, Type(1));
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                detail::is_container_of<V,stride_type>::value>>
        void reset(const U& len, row_view<const pointer> ptr,
                   matrix_view<const len_type> idx, const V& stride)
        {
            unsigned num_idx = ptr.length();
            MARRAY_ASSERT(num_idx > 0);
            MARRAY_ASSERT(idx.length(0) == num_idx);

            unsigned total_dim = len.size();
            unsigned idx_dim = idx.length(1);
            unsigned dense_dim = total_dim - idx_dim;
            MARRAY_ASSERT(total_dim > idx_dim);
            MARRAY_ASSERT(idx_dim > 0 || num_idx == 1);

            data_.reset(ptr);
            idx_.reset(idx);
            dense_len_.assign(len.begin(), std::next(len.begin(),dense_dim));
            idx_len_.assign(std::next(len.begin(),dense_dim), len.end());
            dense_stride_.assign(stride.begin(), stride.end());
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

            for (len_type i = 0;i < num_indices();i++)
            {
                std::copy_n(idx_[i].data(), ndim, indices.data());
                detail::call(std::forward<Func>(f),
                             View(dense_len_, const_cast<Ptr>(data_[i]), dense_stride_),
                             indices);
            }
        }

        template <typename View, typename Func, unsigned... I>
        void for_each_index(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            constexpr unsigned NDim = sizeof...(I);
            typedef typename View::pointer Ptr;

            MARRAY_ASSERT(NDim == indexed_dimension());

            for (len_type i = 0;i < num_indices();i++)
            {
                detail::call(std::forward<Func>(f),
                             View(dense_len_, const_cast<Ptr>(data_[i]), dense_stride_),
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

            for (len_type i = 0;i < num_indices();i++)
            {
                for (unsigned j = 0;j < indexed_ndim;j++)
                    indices[dense_ndim+j] = idx_[i][j];

                Ptr ptr = const_cast<Ptr>(data_[i]);
                for (bool done = false;!done;)
                {
                    detail::call(std::forward<Func>(f), *ptr, indices);

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

            MARRAY_ASSERT(DenseNDim == dense_dimension());
            MARRAY_ASSERT(IdxNDim == indexed_dimension());

            miterator<DenseNDim,1> it(dense_len_, dense_stride_);

            for (len_type i = 0;i < num_indices();i++)
            {
                Ptr ptr = const_cast<Ptr>(data_[i]);
                while (it.next(ptr)) detail::call(std::forward<Func>(f), *ptr,
                                                  it.position()[I]..., idx_[i][J]...);
            }
        }

        template <typename U, typename D, bool O>
        void copy(const indexed_varray_base<U, D, O>& other) const
        {
            MARRAY_ASSERT(lengths() == other.lengths());
            MARRAY_ASSERT(num_indices() == other.num_indices());
            MARRAY_ASSERT(dense_dimension() == other.dense_dimension());

            bool contiguous;
            stride_type size;
            std::tie(contiguous, size) =
                varray_view<Type>::is_contiguous(dense_len_, dense_stride_);

            if (contiguous && dense_strides() == other.dense_strides())
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
                auto it = make_iterator(dense_lengths(), dense_strides(),
                                        other.dense_strides());

                for (len_type i = 0;i < num_indices();i++)
                {
                    MARRAY_ASSERT(indices(i) == other.indices(i));

                    pointer a = const_cast<pointer>(data(i));
                    auto b = other.data(i);

                    while (it.next(a, b)) *a = *b;
                }
            }
        }

        void copy(const Type& value) const
        {
            bool contiguous;
            stride_type size;
            std::tie(contiguous, size) =
                varray_view<Type>::is_contiguous(dense_len_, dense_stride_);

            if (contiguous)
            {
                pointer a = const_cast<pointer>(data(0));

                std::fill_n(a, size*num_indices(), value);
            }
            else
            {
                auto it = make_iterator(dense_lengths(), dense_strides());

                for (len_type i = 0;i < num_indices();i++)
                {
                    pointer a = const_cast<pointer>(data(i));
                    while (it.next(a)) *a = value;
                }
            }
        }

        void swap(indexed_varray_base& other)
        {
            using std::swap;
            swap(data_,         other.data_);
            swap(idx_,          other.idx_);
            swap(dense_len_,    other.dense_len_);
            swap(idx_len_,      other.idx_len_);
            swap(dense_stride_, other.dense_stride_);
        }

    public:

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const indexed_varray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const indexed_varray_base<U, D, O>& other)
        {
            copy(other);
            return static_cast<Derived&>(*this);
        }

        template <typename U, typename D, bool O, bool O_=Owner,
            typename=detail::enable_if_t<!O_ && std::is_assignable<reference,U>::value>>
        const Derived& operator=(const indexed_varray_base<U, D, O>& other) const
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

        indexed_varray_view<const Type> cview() const
        {
            return const_cast<indexed_varray_base&>(*this).view();
        }

        indexed_varray_view<ctype> view() const
        {
            return const_cast<indexed_varray_base&>(*this).view();
        }

        indexed_varray_view<Type> view()
        {
            return *this;
        }

        friend indexed_varray_view<const Type> cview(const indexed_varray_base& x)
        {
            return x.view();
        }

        friend indexed_varray_view<ctype> view(const indexed_varray_base& x)
        {
            return x.view();
        }

        friend indexed_varray_view<Type> view(indexed_varray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        varray_view<ctype> operator[](len_type idx) const
        {
            return const_cast<indexed_varray_base&>(*this)[idx];
        }

        varray_view<Type> operator[](len_type idx)
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            return {dense_len_, data_[idx], dense_stride_};
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_index(Func&& f) const
        {
            for_each_index<varray_view<ctype>>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_index(Func&& f)
        {
            for_each_index<varray_view<Type>>(std::forward<Func>(f));
        }

        template <unsigned DenseNDim, unsigned IdxNDim, typename Func>
        void for_each_index(Func&& f) const
        {
            MARRAY_ASSERT(DenseNDim == dense_dimension());
            for_each_index<marray_view<ctype, DenseNDim>>(std::forward<Func>(f),
                detail::static_range<unsigned, IdxNDim>{});
        }

        template <unsigned DenseNDim, unsigned IdxNDim, typename Func>
        void for_each_index(Func&& f)
        {
            MARRAY_ASSERT(DenseNDim == dense_dimension());
            for_each_index<marray_view<Type, DenseNDim>>(std::forward<Func>(f),
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

        len_type index(len_type idx, len_type dim) const
        {
            MARRAY_ASSERT(0 <= idx && idx < num_indices());
            MARRAY_ASSERT(dim < indexed_dimension());
            return idx_[idx][dim];
        }

        len_type dense_length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dense_dimension());
            return dense_len_[dim];
        }

        const len_vector& dense_lengths() const
        {
            return dense_len_;
        }

        len_type indexed_length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < indexed_dimension());
            return idx_len_[dim];
        }

        const len_vector& indexed_lengths() const
        {
            return idx_len_;
        }

        len_type length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dimension());

            if (dim < dense_dimension()) return dense_length(dim);
            else return indexed_length(dim - dense_dimension());
        }

        len_vector lengths() const
        {
            auto len = dense_len_;
            len.insert(len.end(), idx_len_.begin(), idx_len_.end());
            return len;
        }

        len_type num_indices() const
        {
            return idx_.length(0);
        }

        stride_type dense_stride(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dense_dimension());
            return dense_stride_[dim];
        }

        const stride_vector& dense_strides() const
        {
            return dense_stride_;
        }

        unsigned dimension() const
        {
            return dense_dimension() + indexed_dimension();
        }

        unsigned dense_dimension() const
        {
            return dense_len_.size();
        }

        unsigned indexed_dimension() const
        {
            return idx_.length(1);
        }
};

}

#endif
