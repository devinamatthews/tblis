#ifndef _MARRAY_VARRAY_BASE_HPP_
#define _MARRAY_VARRAY_BASE_HPP_

#include "marray_base.hpp"
#include "viterator.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class varray_base
{
    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
    template <typename, typename, bool> friend class varray_base;
    template <typename> friend class varray_view;
    template <typename, typename> friend class varray;

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
        len_vector len_;
        stride_vector stride_;
        pointer data_ = nullptr;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            data_ = nullptr;
            len_.clear();
            stride_.clear();
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename varray_base<U, D, O>::cptr,pointer>>
        void reset(const varray_base<U, D, O>& other)
        {
            data_ = other.data_;
            len_ = other.len_;
            stride_ = other.stride_;
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename varray_base<U, D, O>::pointer,pointer>>
        void reset(varray_base<U, D, O>& other)
        {
            data_ = other.data_;
            len_ = other.len_;
            stride_ = other.stride_;
        }

        template <typename U, int N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::cptr,pointer>>
        void reset(const marray_base<U, N, D, O>& other)
        {
            data_ = other.data_;
            len_.assign(other.lengths().begin(), other.lengths().end());
            stride_.assign(other.strides().begin(), other.strides().end());
        }

        template <typename U, int N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::pointer,pointer>>
        void reset(marray_base<U, N, D, O>& other)
        {
            data_ = other.data_;
            len_.assign(other.lengths().begin(), other.lengths().end());
            stride_.assign(other.strides().begin(), other.strides().end());
        }

        void reset(detail::array_1d<len_type> len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, strides(len, layout));
        }

        void reset(detail::array_1d<len_type> len, pointer ptr, detail::array_1d<stride_type> stride)
        {
            MARRAY_ASSERT(len.size() > 0);
            MARRAY_ASSERT(len.size() == stride.size());
            data_ = ptr;
            len.slurp(len_);
            stride.slurp(stride_);
        }

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename Ptr, typename Func>
        void for_each_element(Func&& f) const
        {
            viterator<1> it(len_, stride_);
            Ptr ptr = const_cast<Ptr>(data_);
            while (it.next(ptr)) detail::call(std::forward<Func>(f), *ptr, it.position());
        }

        template <typename Ptr, typename Func, int... I>
        void for_each_element(Func&& f, detail::integer_sequence<int, I...>) const
        {
            miterator<sizeof...(I), 1> it(len_, stride_);
            Ptr ptr = const_cast<Ptr>(data_);
            while (it.next(ptr)) detail::call(std::forward<Func>(f), *ptr, it.position()[I]...);
        }

        template <int Dim>
        void get_slice(pointer&, len_vector&,
                       stride_vector&) const {}

        template <int Dim, typename... Args>
        void get_slice(pointer& ptr, len_vector& len,
                       stride_vector& stride,
                       len_type arg, Args&&... args) const
        {
            MARRAY_ASSERT(arg >= 0 && arg < len_[Dim]);
            ptr += arg*stride_[Dim];
            get_slice<Dim+1>(ptr, len, stride, std::forward<Args>(args)...);
        }

        template <int Dim, typename I, typename... Args>
        void get_slice(pointer& ptr, len_vector& len,
                       stride_vector& stride,
                       const range_t<I>& arg, Args&&... args) const
        {
            MARRAY_ASSERT(arg.front() >= 0);
            MARRAY_ASSERT(arg.size() >= 0);
            MARRAY_ASSERT(arg.front()+arg.size() <= len_[Dim]);
            ptr += arg.front()*stride_[Dim];
            len.push_back(arg.size());
            stride.push_back(arg.step()*stride_[Dim]);
            get_slice<Dim+1>(ptr, len, stride, std::forward<Args>(args)...);
        }

        template <int Dim, typename... Args>
        void get_slice(pointer& ptr, len_vector& len,
                       stride_vector& stride,
                       all_t, Args&&... args) const
        {
            len.push_back(len_[Dim]);
            stride.push_back(stride_[Dim]);
            get_slice<Dim+1>(ptr, len, stride, std::forward<Args>(args)...);
        }

        template <int Dim>
        void get_reference(pointer&) const {}

        template <int Dim, typename... Args>
        void get_reference(pointer& ptr, len_type arg, Args&&... args) const
        {
            MARRAY_ASSERT(arg >= 0 && arg < len_[Dim]);
            ptr += arg*stride_[Dim];
            get_reference<Dim+1>(ptr, std::forward<Args>(args)...);
        }

        template <typename U, typename D, bool O>
        void copy(const varray_base<U, D, O>& other) const
        {
            MARRAY_ASSERT(lengths() == other.lengths());

            pointer a = const_cast<pointer>(data());
            auto b = other.data();

            bool contiguous;
            stride_type size;
            std::tie(contiguous, size) = is_contiguous(len_, stride_);

            if (contiguous && strides() == other.strides())
            {
                std::copy_n(b, size, a);
            }
            else
            {
                auto it = make_iterator(lengths(), strides(), other.strides());
                while (it.next(a, b)) *a = *b;
            }
        }

        template <typename U, int N, typename D, bool O>
        void copy(const marray_base<U, N, D, O>& other) const
        {
            MARRAY_ASSERT(dimension() == N);
            MARRAY_ASSERT(std::equal(lengths().begin(), lengths().end(),
                                     other.lengths().begin()));

            pointer a = const_cast<pointer>(data());
            auto b = other.data();

            bool contiguous;
            stride_type size;
            std::tie(contiguous, size) = is_contiguous(len_, stride_);

            if (contiguous && std::equal(strides().begin(), strides().end(),
                                         other.strides().begin()))
            {
                std::copy_n(b, size, a);
            }
            else
            {
                auto it = make_iterator(lengths(), strides(), other.strides());
                while (it.next(a, b)) *a = *b;
            }
        }

        void copy(const Type& value) const
        {
            pointer a = const_cast<pointer>(data());

            bool contiguous;
            stride_type size;
            std::tie(contiguous, size) = is_contiguous(len_, stride_);

            if (contiguous)
            {
                std::fill_n(a, size, value);
            }
            else
            {
                auto it = make_iterator(lengths(), strides());
                while (it.next(a)) *a = value;
            }
        }

        void swap(varray_base& other)
        {
            using std::swap;
            swap(data_, other.data_);
            swap(len_, other.len_);
            swap(stride_, other.stride_);
        }

    public:

        /***********************************************************************
         *
         * Static helper functions
         *
         **********************************************************************/

        static stride_vector strides(detail::array_1d<len_type> len_, layout layout = DEFAULT)
        {
            len_vector len;
            len_.slurp(len);

            MARRAY_ASSERT(len.size() > 0);

            int ndim = len.size();
            stride_vector stride(ndim);

            if (layout == ROW_MAJOR)
            {
                /*
                 * Some monkeying around has to be done to support len as a
                 * std::initializer_list (or other forward iterator range)
                 */
                auto it = len.begin(); ++it;
                std::copy_n(it, ndim-1, stride.begin());
                stride[ndim-1] = 1;
                for (auto i : reversed_range(ndim-1))
                    stride[i] *= stride[i+1];
            }
            else
            {
                stride[0] = 1;
                auto it = len.begin();
                for (auto i : range(1,ndim))
                {
                    stride[i] = stride[i-1]*(*it);
                    ++it;
                }
            }

            return stride;
        }

        static stride_type size(detail::array_1d<len_type> len_)
        {
            len_vector len;
            len_.slurp(len);
            MARRAY_ASSERT(len.size() > 0);
            stride_type s = 1;
            for (auto& l : len) s *= l;
            return s;
        }

        static std::pair<bool,stride_type> is_contiguous(detail::array_1d<len_type> len_,
                                                         detail::array_1d<stride_type> stride_)
        {
            len_vector len;
            len_.slurp(len);

            stride_vector stride;
            stride_.slurp(stride);

            int ndim = len.size();
            MARRAY_ASSERT(ndim > 0);
            MARRAY_ASSERT(len.size() == stride.size());

            if (len.size() == 1) return std::make_pair(true, *len.begin());

            auto len_it = len.begin();
            auto stride_it = stride.begin();

            len_type len0 = *len_it;
            stride_type stride0 = *stride_it;

            ++len_it;
            ++stride_it;

            len_type len1 = *len_it;
            stride_type stride1 = *stride_it;

            stride_type size = len0;

            if (stride0 < stride1)
            {
                for (auto i : range(1,ndim))
                {
                    size *= len1;

                    if (stride1 != stride0*len0)
                        return std::make_pair(false, stride_type());

                    if (i < ndim-1)
                    {
                        len0 = len1;
                        stride0 = stride1;
                        len1 = *(++len_it);
                        stride1 = *(++stride_it);
                    }
                }
            }
            else
            {
                for (auto i : range(1,ndim))
                {
                    size *= len1;

                    if (stride0 != stride1*len1)
                        return std::make_pair(false, stride_type());

                    if (i < ndim-1)
                    {
                        len0 = len1;
                        stride0 = stride1;
                        len1 = *(++len_it);
                        stride1 = *(++stride_it);
                    }
                }
            }

            return std::make_pair(true, size);
        }

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const varray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const varray_base<U, D, O>& other)
        {
            copy(other);
            return static_cast<Derived&>(*this);
        }

        template <typename U, typename D, bool O, bool O_=Owner,
            typename=detail::enable_if_t<!O_ && std::is_assignable<reference,U>::value>>
        const Derived& operator=(const varray_base<U, D, O>& other) const
        {
            copy(other);
            return static_cast<const Derived&>(*this);
        }

        template <typename U, int N, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const marray_base<U, N, D, O>& other)
        {
            copy(other);
            return static_cast<Derived&>(*this);
        }

        template <typename U, int N, typename D, bool O, bool O_=Owner,
            typename=detail::enable_if_t<!O_ && std::is_assignable<reference,U>::value>>
        const Derived& operator=(const marray_base<U, N, D, O>& other) const
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

        varray_view<const Type> cview() const
        {
            return *this;
        }

        varray_view<ctype> view() const
        {
            return *this;
        }

        varray_view<Type> view()
        {
            return *this;
        }

        friend varray_view<const Type> cview(const varray_base& x)
        {
            return x.cview();
        }

        friend varray_view<ctype> view(const varray_base& x)
        {
            return x.view();
        }

        friend varray_view<Type> view(varray_base& x)
        {
            return x.view();
        }

        template <int NDim>
        marray_view<ctype, NDim> fix() const
        {
            return const_cast<varray_base&>(*this).fix<NDim>();
        }

        template <int NDim>
        marray_view<Type, NDim> fix()
        {
            MARRAY_ASSERT(NDim == dimension());

            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;
            std::copy_n(len_.begin(), NDim, len.begin());
            std::copy_n(stride_.begin(), NDim, stride.begin());

            return {len, data_, stride};
        }

        /***********************************************************************
         *
         * Shifting
         *
         **********************************************************************/

        varray_view<ctype> shifted(detail::array_1d<len_type> n) const
        {
            return const_cast<varray_base&>(*this).shifted(n);
        }

        varray_view<Type> shifted(detail::array_1d<len_type> n)
        {
            MARRAY_ASSERT(n.size() == dimension());
            varray_view<Type> r(*this);
            r.shift(n);
            return r;
        }

        varray_view<ctype> shifted(int dim, len_type n) const
        {
            return const_cast<varray_base&>(*this).shifted(dim, n);
        }

        varray_view<Type> shifted(int dim, len_type n)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            varray_view<Type> r(*this);
            r.shift(dim, n);
            return r;
        }

        varray_view<ctype> shifted_down(int dim) const
        {
            return const_cast<varray_base&>(*this).shifted_down(dim);
        }

        varray_view<Type> shifted_down(int dim)
        {
            return shifted(dim, len_[dim]);
        }

        varray_view<ctype> shifted_up(int dim) const
        {
            return const_cast<varray_base&>(*this).shifted_up(dim);
        }

        varray_view<Type> shifted_up(int dim)
        {
            return shifted(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        varray_view<ctype> permuted(detail::array_1d<int> perm) const
        {
            return const_cast<varray_base&>(*this).permuted(perm);
        }

        varray_view<Type> permuted(detail::array_1d<int> perm)
        {
            varray_view<Type> r(*this);
            r.permute(perm);
            return r;
        }

        /***********************************************************************
         *
         * Dimension change
         *
         **********************************************************************/

        varray_view<ctype> lowered(detail::array_1d<int> split) const
        {
            return const_cast<varray_base&>(*this).lowered(split);
        }

        varray_view<Type> lowered(detail::array_1d<int> split)
        {
            varray_view<Type> r(*this);
            r.lower(split);
            return r;
        }

        /***********************************************************************
         *
         * Reversal
         *
         **********************************************************************/

        varray_view<ctype> reversed() const
        {
            return const_cast<varray_base&>(*this).reversed();
        }

        varray_view<Type> reversed()
        {
            varray_view<Type> r(*this);
            r.reverse();
            return r;
        }

        varray_view<ctype> reversed(int dim) const
        {
            return const_cast<varray_base&>(*this).reversed(dim);
        }

        varray_view<Type> reversed(int dim)
        {
            varray_view<Type> r(*this);
            r.reverse(dim);
            return r;
        }

        /***********************************************************************
         *
         * Slicing
         *
         **********************************************************************/

        const_reference cfront() const
        {
            return const_cast<varray_base&>(*this).front();
        }

        cref front() const
        {
            return const_cast<varray_base&>(*this).front();
        }

        reference front()
        {
            MARRAY_ASSERT(dimension() == 1);
            MARRAY_ASSERT(len_[0] > 0);
            return data_[0];
        }

        varray_view<const Type> cfront(int dim) const
        {
            return const_cast<varray_base&>(*this).front(dim);
        }

        varray_view<ctype> front(int dim) const
        {
            return const_cast<varray_base&>(*this).front(dim);
        }

        varray_view<Type> front(int dim)
        {
            MARRAY_ASSERT(dimension() > 1);
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(len_[dim] > 0);

            len_vector len(dimension()-1);
            stride_vector stride(dimension()-1);

            std::copy_n(len_.begin(), dim, len.begin());
            std::copy_n(len_.begin()+dim+1, dimension()-dim-1, len.begin()+dim);
            std::copy_n(stride_.begin(), dim, stride.begin());
            std::copy_n(stride_.begin()+dim+1, dimension()-dim-1, stride.begin()+dim);

            return {len, data_, stride};
        }

        const_reference cback() const
        {
            return const_cast<varray_base&>(*this).back();
        }

        cref back() const
        {
            return const_cast<varray_base&>(*this).back();
        }

        reference back()
        {
            MARRAY_ASSERT(dimension() == 1);
            MARRAY_ASSERT(len_[0] > 0);
            return data_[(len_[0]-1)*stride_[0]];
        }

        varray_view<const Type> cback(int dim) const
        {
            return const_cast<varray_base&>(*this).back(dim);
        }

        varray_view<ctype> back(int dim) const
        {
            return const_cast<varray_base&>(*this).back(dim);
        }

        varray_view<Type> back(int dim)
        {
            MARRAY_ASSERT(dimension() > 1);
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(len_[dim] > 0);

            len_vector len(dimension()-1);
            stride_vector stride(dimension()-1);

            std::copy_n(len_.begin(), dim, len.begin());
            std::copy_n(len_.begin()+dim+1, dimension()-dim-1, len.begin()+dim);
            std::copy_n(stride_.begin(), dim, stride.begin());
            std::copy_n(stride_.begin()+dim+1, dimension()-dim-1, stride.begin()+dim);

            return {len, data_+(len_[dim]-1)*stride_[dim], stride};
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        cref operator()(detail::array_1d<len_type> idx_) const
        {
            return const_cast<varray_base&>(*this)(idx_);
        }

        reference operator()(detail::array_1d<len_type> idx_)
        {
            MARRAY_ASSERT(idx_.size() == dimension());

            len_vector idx;
            idx_.slurp(idx);

            auto ptr = data();
            for (auto i : range(dimension()))
                ptr += idx[i]*stride(i);

            return *ptr;
        }

        template <typename... Args>
        detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                            !detail::are_convertible<len_type, Args...>::value,
                            varray_view<ctype>>
        operator()(Args&&... args) const
        {
            return const_cast<varray_base&>(*this)(args...);
        }

        template <typename... Args>
        detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                            !detail::are_convertible<len_type, Args...>::value,
                            varray_view<Type>>
        operator()(Args&&... args)
        {
            MARRAY_ASSERT(sizeof...(Args) == dimension());

            pointer ptr = data();
            len_vector len;
            stride_vector stride;

            get_slice<0>(ptr, len, stride, std::forward<Args>(args)...);

            return {len, ptr, stride};
        }

        template <typename... Args>
        detail::enable_if_t<detail::are_convertible<len_type, Args...>::value,
                            cref>
        operator()(Args&&... args) const
        {
            return const_cast<varray_base&>(*this)(args...);
        }

        template <typename... Args>
        detail::enable_if_t<detail::are_convertible<len_type, Args...>::value,
                            reference>
        operator()(Args&&... args)
        {
            MARRAY_ASSERT(sizeof...(Args) == dimension());
            pointer ptr = data();
            get_reference<0>(ptr, std::forward<Args>(args)...);
            return *ptr;
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<cptr>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<pointer>(std::forward<Func>(f));
        }

        template <int NDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<cptr>(std::forward<Func>(f), detail::static_range<int, NDim>{});
        }

        template <int NDim, typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<pointer>(std::forward<Func>(f), detail::static_range<int, NDim>{});
        }

        /***********************************************************************
         *
         * Basic getters
         *
         **********************************************************************/

        const_pointer cdata() const
        {
            return data_;
        }

        cptr data() const
        {
            return data_;
        }

        pointer data()
        {
            return data_;
        }

        len_type length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return len_[dim];
        }

        const len_vector& lengths() const
        {
            return len_;
        }

        stride_type stride(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return stride_[dim];
        }

        const stride_vector& strides() const
        {
            return stride_;
        }

        int dimension() const
        {
            return len_.size();
        }

        friend std::ostream& operator<<(std::ostream& os, const varray_base& x)
        {
            auto ndim = x.dimension();

            for (auto i : range(ndim-1))
            {
                for (auto j : range(i)) os << ' ';
                os << "{\n";
            }

            len_vector idx(ndim-1);
            auto data = x.data_;

            for (bool done = false;!done;)
            {
                for (auto i : range(ndim-1)) os << ' ';
                os << '{';
                auto n = x.len_[ndim-1];
                for (auto i : range(n-1))
                    os << data[i*x.stride_[ndim-1]] << ", ";
                os << data[(n-1)*x.stride_[ndim-1]] << "}";

                for (auto i : reversed_range(ndim-1))
                {
                    idx[i]++;
                    data += x.stride_[i];

                    if (idx[i] >= x.len_[i])
                    {
                        data -= idx[i]*x.stride_[i];
                        idx[i] = 0;
                        os << "\n";
                        for (auto j : range(i)) os << ' ';
                        os << '}';
                        if (i == 0) done = true;
                    }
                    else
                    {
                        os << ",\n";
                        for (auto j : range(i+1,ndim-1))
                        {
                            for (auto k : range(j)) os << ' ';
                            os << "{\n";
                        }
                        break;
                    }
                }

                if (ndim == 1) break;
            }

            return os;
        }
};

template <int NDim, typename Type, typename Derived>
marray_view<Type, NDim> fix(const varray_base<Type, Derived, false>& x)
{
    return x.template fix<NDim>();
}

template <int NDim, typename Type, typename Derived>
marray_view<const Type, NDim> fix(const varray_base<Type, Derived, true>& x)
{
    return x.template fix<NDim>();
}

template <int NDim, typename Type, typename Derived>
marray_view<Type, NDim> fix(varray_base<Type, Derived, true>& x)
{
    return x.template fix<NDim>();
}

template <int NDim, typename Type, typename Derived>
marray_view<Type, NDim> fix(varray_base<Type, Derived, true>&& x)
{
    return x.template fix<NDim>();
}

template <int NDim1, typename Type, int NDim2, typename Derived>
marray_view<Type, NDim1> fix(const marray_base<Type, NDim2, Derived, false>& x)
{
    static_assert(NDim1 == NDim2, "Dimensions must be equal");
    return x;
}

template <int NDim1, typename Type, int NDim2, typename Derived>
marray_view<const Type, NDim1> fix(const marray_base<Type, NDim2, Derived, true>& x)
{
    static_assert(NDim1 == NDim2, "Dimensions must be equal");
    return x;
}

template <int NDim1, typename Type, int NDim2, typename Derived>
marray_view<Type, NDim1> fix(marray_base<Type, NDim2, Derived, true>& x)
{
    static_assert(NDim1 == NDim2, "Dimensions must be equal");
    return x;
}

template <int NDim1, typename Type, int NDim2, typename Derived>
marray_view<Type, NDim1> fix(marray_base<Type, NDim2, Derived, true>&& x)
{
    static_assert(NDim1 == NDim2, "Dimensions must be equal");
    return x;
}

//TODO: maybe these should go in marray_base?

template <typename U, int N, typename D>
varray_view<U> vary(const marray_base<U, N, D, false>& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, int N, typename D>
varray_view<const U> vary(const marray_base<U, N, D, true>& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, int N, typename D>
varray_view<const U> vary(const marray_base<U, N, D, true>&& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, int N, typename D>
varray_view<U> vary(marray_base<U, N, D, true>& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, typename D>
varray_view<U> vary(const varray_base<U, D, false>& other)
{
    return other;
}

template <typename U, typename D>
varray_view<const U> vary(const varray_base<U, D, true>& other)
{
    return other;
}

template <typename U, typename D>
varray_view<U> vary(varray_base<U, D, true>& other)
{
    return other;
}

template <typename U, typename D>
varray_view<U> vary(varray_base<U, D, true>&& other)
{
    return other;
}

template <typename U, int N, int I, typename... Ds>
varray_view<U> vary(const marray_slice<U, N, I, Ds...>& other)
{
    return vary(other.view());
}

}

#endif
