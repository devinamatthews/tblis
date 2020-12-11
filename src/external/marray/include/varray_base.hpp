#ifndef _MARRAY_VARRAY_BASE_HPP_
#define _MARRAY_VARRAY_BASE_HPP_

#include "marray_base.hpp"
#include "viterator.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class varray_base
{
    template <typename, unsigned, typename, bool> friend class marray_base;
    template <typename, unsigned> friend class marray_view;
    template <typename, unsigned, typename> friend class marray;
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

        template <typename U, unsigned N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::cptr,pointer>>
        void reset(const marray_base<U, N, D, O>& other)
        {
            data_ = other.data_;
            len_.assign(other.lengths().begin(), other.lengths().end());
            stride_.assign(other.strides().begin(), other.strides().end());
        }

        template <typename U, unsigned N, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, N, D, O>::pointer,pointer>>
        void reset(marray_base<U, N, D, O>& other)
        {
            data_ = other.data_;
            len_.assign(other.lengths().begin(), other.lengths().end());
            stride_.assign(other.strides().begin(), other.strides().end());
        }

        void reset(std::initializer_list<len_type> len, pointer ptr, layout layout = DEFAULT)
        {
            reset<std::initializer_list<len_type>>(len, ptr, layout);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        void reset(const U& len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, strides(len, layout));
        }

        void reset(std::initializer_list<len_type> len, pointer ptr,
                   std::initializer_list<stride_type> stride)
        {
            reset<std::initializer_list<len_type>,
                  std::initializer_list<stride_type>>(len, ptr, stride);
        }

        template <typename U, typename V, typename=detail::enable_if_t<
            detail::is_container_of<U,len_type>::value &&
            detail::is_container_of<V,stride_type>::value>>
        void reset(const U& len, pointer ptr, const V& stride)
        {
            MARRAY_ASSERT(len.size() > 0);
            MARRAY_ASSERT(len.size() == stride.size());
            data_ = ptr;
            len_.assign(len.begin(), len.end());
            stride_.assign(stride.begin(), stride.end());
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

        template <typename Ptr, typename Func, unsigned... I>
        void for_each_element(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            miterator<sizeof...(I), 1> it(len_, stride_);
            Ptr ptr = const_cast<Ptr>(data_);
            while (it.next(ptr)) detail::call(std::forward<Func>(f), *ptr, it.position()[I]...);
        }

        template <unsigned Dim>
        void get_slice(pointer&, len_vector&,
                       stride_vector&) const {}

        template <unsigned Dim, typename... Args>
        void get_slice(pointer& ptr, len_vector& len,
                       stride_vector& stride,
                       len_type arg, Args&&... args) const
        {
            MARRAY_ASSERT(arg >= 0 && arg < len_[Dim]);
            ptr += arg*stride_[Dim];
            get_slice<Dim+1>(ptr, len, stride, std::forward<Args>(args)...);
        }

        template <unsigned Dim, typename I, typename... Args>
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

        template <unsigned Dim, typename... Args>
        void get_slice(pointer& ptr, len_vector& len,
                       stride_vector& stride,
                       all_t, Args&&... args) const
        {
            len.push_back(len_[Dim]);
            stride.push_back(stride_[Dim]);
            get_slice<Dim+1>(ptr, len, stride, std::forward<Args>(args)...);
        }

        template <unsigned Dim>
        void get_reference(pointer&) const {}

        template <unsigned Dim, typename... Args>
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

        template <typename U, unsigned N, typename D, bool O>
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

        static stride_vector
        strides(std::initializer_list<len_type> len, layout layout = DEFAULT)
        {
            return strides<std::initializer_list<len_type>>(len, layout);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        static stride_vector
        strides(const U& len, layout layout = DEFAULT)
        {
            MARRAY_ASSERT(len.size() > 0);

            auto ndim = len.size();
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
                for (unsigned i = ndim;i --> 1;)
                {
                    stride[i-1] *= stride[i];
                }
            }
            else
            {
                stride[0] = 1;
                auto it = len.begin();
                for (unsigned i = 1;i < ndim;i++)
                {
                    stride[i] = stride[i-1]*(*it);
                    ++it;
                }
            }

            return stride;
        }

        stride_type size(std::initializer_list<len_type> len)
        {
            return size<std::initializer_list<len_type>>(len);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        static stride_type size(const U& len)
        {
            MARRAY_ASSERT(len.size() > 0);
            stride_type s = 1;
            for (auto& l : len) s *= l;
            return s;
        }

        static std::pair<bool,stride_type> is_contiguous(std::initializer_list<len_type> len,
                                                         std::initializer_list<stride_type> stride)
        {
            return is_contiguous<std::initializer_list<len_type>,
                                 std::initializer_list<stride_type>,void>(len, stride);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                detail::is_container_of<V,stride_type>::value>>
        static std::pair<bool,stride_type> is_contiguous(const U& len, const V& stride)
        {
            MARRAY_ASSERT(len.size() > 0);
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
                for (unsigned i = 1;i < len.size();i++)
                {
                    size *= len1;

                    if (stride1 != stride0*len0)
                        return std::make_pair(false, stride_type());

                    if (i < len.size()-1)
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
                for (unsigned i = 1;i < len.size();i++)
                {
                    size *= len1;

                    if (stride0 != stride1*len1)
                        return std::make_pair(false, stride_type());

                    if (i < len.size()-1)
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

        template <typename U, unsigned N, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const marray_base<U, N, D, O>& other)
        {
            copy(other);
            return static_cast<Derived&>(*this);
        }

        template <typename U, unsigned N, typename D, bool O, bool O_=Owner,
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

        template <unsigned NDim>
        marray_view<ctype, NDim> fix() const
        {
            return const_cast<varray_base&>(*this).fix<NDim>();
        }

        template <unsigned NDim>
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

        varray_view<ctype> shifted(std::initializer_list<len_type> n) const
        {
            return const_cast<varray_base&>(*this).shifted(n);
        }

        varray_view<Type> shifted(std::initializer_list<len_type> n)
        {
            return shifted<std::initializer_list<len_type>>(n);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        varray_view<ctype> shifted(const U& n) const
        {
            return const_cast<varray_base&>(*this).shifted(n);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        varray_view<Type> shifted(const U& n)
        {
            MARRAY_ASSERT(n.size() == dimension());
            varray_view<Type> r(*this);
            r.shift(n);
            return r;
        }

        varray_view<ctype> shifted(unsigned dim, len_type n) const
        {
            return const_cast<varray_base&>(*this).shifted(dim, n);
        }

        varray_view<Type> shifted(unsigned dim, len_type n)
        {
            MARRAY_ASSERT(dim < dimension());
            varray_view<Type> r(*this);
            r.shift(dim, n);
            return r;
        }

        varray_view<ctype> shifted_down(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).shifted_down(dim);
        }

        varray_view<Type> shifted_down(unsigned dim)
        {
            return shifted(dim, len_[dim]);
        }

        varray_view<ctype> shifted_up(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).shifted_up(dim);
        }

        varray_view<Type> shifted_up(unsigned dim)
        {
            return shifted(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        varray_view<ctype> permuted(std::initializer_list<unsigned> perm) const
        {
            return const_cast<varray_base&>(*this).permuted(perm);
        }

        varray_view<Type> permuted(std::initializer_list<unsigned> perm)
        {
            return permuted<std::initializer_list<unsigned>>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<ctype> permuted(const U& perm) const
        {
            return const_cast<varray_base&>(*this).permuted(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<Type> permuted(const U& perm)
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

        varray_view<ctype> lowered(std::initializer_list<unsigned> split) const
        {
            return const_cast<varray_base&>(*this).lowered(split);
        }

        varray_view<Type> lowered(std::initializer_list<unsigned> split)
        {
            return lowered<std::initializer_list<unsigned>>(split);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<ctype> lowered(const U& split) const
        {
            return const_cast<varray_base&>(*this).lowered(split);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<Type> lowered(const U& split)
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

        varray_view<ctype> reversed(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).reversed(dim);
        }

        varray_view<Type> reversed(unsigned dim)
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

        varray_view<const Type> cfront(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).front(dim);
        }

        varray_view<ctype> front(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).front(dim);
        }

        varray_view<Type> front(unsigned dim)
        {
            MARRAY_ASSERT(dimension() > 1);
            MARRAY_ASSERT(dim < dimension());
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

        varray_view<const Type> cback(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).back(dim);
        }

        varray_view<ctype> back(unsigned dim) const
        {
            return const_cast<varray_base&>(*this).back(dim);
        }

        varray_view<Type> back(unsigned dim)
        {
            MARRAY_ASSERT(dimension() > 1);
            MARRAY_ASSERT(dim < dimension());
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

        template <unsigned NDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<cptr>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <unsigned NDim, typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<pointer>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
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

        len_type length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dimension());
            return len_[dim];
        }

        const len_vector& lengths() const
        {
            return len_;
        }

        stride_type stride(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dimension());
            return stride_[dim];
        }

        const stride_vector& strides() const
        {
            return stride_;
        }

        unsigned dimension() const
        {
            return static_cast<unsigned>(len_.size());
        }

        friend std::ostream& operator<<(std::ostream& os, const varray_base& x)
        {
            unsigned ndim = x.dimension();

            for (unsigned i = 0;i < ndim-1;i++)
            {
                for (unsigned j = 0;j < i;j++) os << ' ';
                os << "{\n";
            }

            len_vector idx(ndim-1);
            auto data = x.data_;

            for (bool done = false;!done;)
            {
                for (unsigned i = 0;i < ndim-1;i++) os << ' ';
                os << '{';
                len_type n = x.len_[ndim-1];
                for (len_type i = 0;i < n-1;i++)
                    os << data[i*x.stride_[ndim-1]] << ", ";
                os << data[(n-1)*x.stride_[ndim-1]] << "}";

                for (unsigned i = ndim-1;i --> 0;)
                {
                    idx[i]++;
                    data += x.stride_[i];

                    if (idx[i] >= x.len_[i])
                    {
                        data -= idx[i]*x.stride_[i];
                        idx[i] = 0;
                        os << "\n";
                        for (unsigned j = 0;j < i;j++) os << ' ';
                        os << '}';
                        if (i == 0) done = true;
                    }
                    else
                    {
                        os << ",\n";
                        for (unsigned j = i+1;j < ndim-1;j++)
                        {
                            for (unsigned k = 0;k < j;k++) os << ' ';
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

template <unsigned NDim, typename Type, typename Derived>
marray_view<Type, NDim> fix(const varray_base<Type, Derived, false>& x)
{
    return x.template fix<NDim>();
}

template <unsigned NDim, typename Type, typename Derived>
marray_view<const Type, NDim> fix(const varray_base<Type, Derived, true>& x)
{
    return x.template fix<NDim>();
}

template <unsigned NDim, typename Type, typename Derived>
marray_view<Type, NDim> fix(varray_base<Type, Derived, true>& x)
{
    return x.template fix<NDim>();
}

//TODO: maybe these should go in marray_base?

template <typename U, unsigned N, typename D>
varray_view<U> vary(const marray_base<U, N, D, false>& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, unsigned N, typename D>
varray_view<const U> vary(const marray_base<U, N, D, true>& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, unsigned N, typename D>
varray_view<U> vary(marray_base<U, N, D, true>& other)
{
    len_vector len{other.lengths().begin(), other.lengths().end()};
    stride_vector stride{other.strides().begin(), other.strides().end()};
    return {len, other.data(), stride};
}

template <typename U, unsigned N, unsigned I, typename... Ds>
varray_view<U> vary(const marray_slice<U, N, I, Ds...>& other)
{
    return vary(other.view());
}

}

#endif
