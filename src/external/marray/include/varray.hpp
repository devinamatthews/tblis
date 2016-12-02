#ifndef _MARRAY_VARRAY_HPP_
#define _MARRAY_VARRAY_HPP_

#include <type_traits>
#include <array>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <utility>

#include "viterator.hpp"
#include "marray.hpp"
#include "utility.hpp"

namespace MArray
{
    template <typename T>
    class const_varray_view;

    template <typename T>
    class varray_view;

    template <typename T, typename Allocator>
    class varray;

    template <typename T> void copy(const_varray_view<T> a, varray_view<T> b);

    template <typename T>
    class const_varray_view
    {
        template <typename T_> friend class const_varray_view;
        template <typename T_> friend class varray_view;
        template <typename T_, typename Allocator_> friend class varray;

        public:
            typedef ssize_t idx_type;
            typedef size_t size_type;
            typedef ptrdiff_t stride_type;
            typedef T value_type;
            typedef T* pointer;
            typedef const T* const_pointer;
            typedef T& reference;
            typedef const T& const_reference;

        protected:
            pointer data_ = nullptr;
            std::vector<idx_type> len_;
            std::vector<stride_type> stride_;

            const_varray_view& operator=(const const_varray_view& other) = delete;

        public:
            static std::vector<stride_type> default_strides(const std::vector<idx_type>& len, Layout layout=DEFAULT)
            {
                return default_strides<idx_type>(len, layout);
            }

            template <typename U>
            static detail::enable_if_integral_t<U,std::vector<stride_type>>
            default_strides(const std::vector<U>& len, Layout layout=DEFAULT)
            {
                std::vector<stride_type> stride(len.size());

                if (stride.empty()) return stride;

                auto ndim = len.size();
                if (layout == ROW_MAJOR)
                {
                    stride[ndim-1] = 1;
                    for (auto i = ndim;i --> 1;)
                    {
                        stride[i-1] = stride[i]*len[i];
                    }
                }
                else
                {
                    stride[0] = 1;
                    for (unsigned i = 1;i < ndim;i++)
                    {
                        stride[i] = stride[i-1]*len[i-1];
                    }
                }

                return stride;
            }

            const_varray_view() {}

            const_varray_view(const const_varray_view<T>& other)
            {
                reset(other);
            }

            const_varray_view(const varray_view<T>& other)
            {
                reset(other);
            }

            template <typename Alloc, typename=
                detail::enable_if_not_integral_t<Alloc>>
            const_varray_view(const varray<T, Alloc>& other)
            {
                reset(other);
            }

            const_varray_view(const_varray_view<T>&& other)
            {
                reset(std::move(other));
            }

            const_varray_view(varray_view<T>&& other)
            {
                reset(std::move(other));
            }

            const_varray_view(const std::vector<idx_type>& len, const_pointer ptr, Layout layout=DEFAULT)
            {
                reset(len, ptr, layout);
            }

            template <typename U, typename=
                detail::enable_if_integral_t<U>>
            const_varray_view(const std::vector<U>& len, const_pointer ptr, Layout layout=DEFAULT)
            {
                reset(len, ptr, layout);
            }

            const_varray_view(const std::vector<idx_type>& len, const_pointer ptr, const std::vector<stride_type>& stride)
            {
                reset(len, ptr, stride);
            }

            template <typename U, typename V, typename=
                detail::enable_if_t<std::is_integral<U>::value &&
                                     std::is_integral<V>::value>>
            const_varray_view(const std::vector<U>& len, const_pointer ptr, const std::vector<V>& stride)
            {
                reset(len, ptr, stride);
            }

            void reset()
            {
                data_ = nullptr;
                len_.clear();
                stride_.clear();
            }

            void reset(const const_varray_view<T>& other)
            {
                data_ = other.data_;
                len_ = other.len_;
                stride_ = other.stride_;
            }

            void reset(const varray_view<T>& other)
            {
                reset(static_cast<const const_varray_view<T>&>(other));
            }

            template <typename Alloc>
            detail::enable_if_not_integral_t<Alloc>
            reset(const varray<T, Alloc>& other)
            {
                reset(static_cast<const const_varray_view<T>&>(other));
            }

            void reset(const_varray_view<T>&& other)
            {
                data_ = other.data_;
                len_ = std::move(other.len_);
                stride_ = std::move(other.stride_);
            }

            void reset(varray_view<T>&& other)
            {
                reset(static_cast<const_varray_view<T>&&>(other));
            }

            void reset(const std::vector<idx_type>& len, const_pointer ptr, Layout layout = DEFAULT)
            {
                reset<idx_type>(len, ptr, layout);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            reset(const std::vector<U>& len, const_pointer ptr, Layout layout = DEFAULT)
            {
                reset(len, ptr, default_strides(len, layout));
            }

            void reset(const std::vector<idx_type>& len, const_pointer ptr, const std::vector<stride_type>& stride)
            {
                reset<idx_type, stride_type>(len, ptr, stride);
            }

            template <typename U, typename V>
            detail::enable_if_t<std::is_integral<U>::value &&
                                 std::is_integral<V>::value>
            reset(const std::vector<U>& len, const_pointer ptr, const std::vector<V>& stride)
            {
                assert(len.size() == stride.size());
                data_ = const_cast<pointer>(ptr);
                len_ = len;
                stride_ = stride;
            }

            void shift(unsigned dim, idx_type n)
            {
                assert(dim < dimension());
                data_ += n*stride_[dim];
            }

            void shift_down(unsigned dim)
            {
                shift(dim, len_[dim]);
            }

            void shift_up(unsigned dim)
            {
                shift(dim, -len_[dim]);
            }

            const_varray_view<T> shifted(unsigned dim, idx_type n) const
            {
                assert(dim < dimension());
                const_varray_view<T> r(*this);
                r.shift(dim, n);
                return r;
            }

            const_varray_view<T> shifted_down(unsigned dim) const
            {
                return shifted(dim, len_[dim]);
            }

            const_varray_view<T> shifted_up(unsigned dim) const
            {
                return shifted(dim, -len_[dim]);
            }

            void permute(const std::vector<unsigned>& perm)
            {
                permute<unsigned>(perm);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            permute(const std::vector<U>& perm)
            {
                assert(perm.size() == dimension());

                std::vector<idx_type> len(dimension());
                std::vector<stride_type> stride(dimension());

                for (unsigned i = 0;i < dimension();i++)
                {
                    assert(0 <= perm[i] && perm[i] < dimension());
                    for (unsigned j = 0;j < i;j++) assert(perm[i] != perm[j]);
                }

                for (unsigned i = 0;i < dimension();i++)
                {
                    len_[i] = len[perm[i]];
                    stride_[i] = stride[perm[i]];
                }
            }

            const_varray_view<T> permuted(const std::vector<unsigned>& perm) const
            {
                return permuted<unsigned>(perm);
            }

            template <typename U>
            detail::enable_if_integral_t<U,const_varray_view<T>>
            permuted(const std::vector<U>& perm) const
            {
                const_varray_view<T> r(*this);
                r.permute(perm);
                return r;
            }

            void lower(const std::vector<unsigned>& split)
            {
                lower<unsigned>(split);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            lower(const std::vector<U>& split)
            {
                assert(split.size() < dimension());

                unsigned newdim = split.size()+1;
                for (unsigned i = 0;i < newdim-1;i++)
                {
                    assert(split[i] <= dimension());
                    if (i != 0) assert(split[i-1] <= split[i]);
                }

                std::vector<idx_type> len = len_;
                std::vector<stride_type> stride = stride_;

                for (unsigned i = 0;i < newdim;i++)
                {
                    unsigned begin = (i == 0 ? 0 : split[i-1]);
                    unsigned end = (i == newdim-1 ? dimension()-1 : split[i]-1);
                    if (begin > end) continue;

                    if (stride[begin] < stride[end])
                    {
                        len_[i] = len[end];
                        stride_[i] = stride[begin];
                        for (auto j = begin;j < end;j++)
                        {
                            assert(stride[j+1] == stride[j]*len[j]);
                            len_[i] *= len[j];
                        }
                    }
                    else
                    {
                        len_[i] = len[end];
                        stride_[i] = stride[end];
                        for (auto j = begin;j < end;j++)
                        {
                            assert(stride[j] == stride[j+1]*len[j+1]);
                            len_[i] *= len[j];
                        }
                    }
                }

                len_.resize(newdim);
                stride_.resize(newdim);
            }

            const_varray_view<T> lowered(const std::vector<unsigned>& split) const
            {
                return lowered<unsigned>(split);
            }

            template <typename U>
            detail::enable_if_integral_t<U,const_varray_view<T>>
            lowered(const std::vector<U>& split) const
            {
                const_varray_view<T> r(*this);
                r.lower(split);
                return r;
            }

            const_reference front() const
            {
                assert(dimension() == 1);
                assert(len_[0] > 0);
                return data_[0];
            }

            const_varray_view<T> front(unsigned dim) const
            {
                assert(dim < dimension());
                assert(len_[dim] > 0);

                std::vector<idx_type> len(dimension()-1);
                std::vector<stride_type> stride(dimension()-1);

                std::copy_n(len_.begin(), dim, len.begin());
                std::copy_n(len_.begin()+dim+1, dimension()-dim-1, len.begin()+dim);
                std::copy_n(stride_.begin(), dim, stride.begin());
                std::copy_n(stride_.begin()+dim+1, dimension()-dim-1, stride.begin()+dim);

                return {len, data_, stride};
            }

            const_reference back() const
            {
                assert(dimension() == 1);
                assert(len_[0] > 0);
                return data_[(len_[0]-1)*stride_[0]];
            }

            const_varray_view<T> back(unsigned dim) const
            {
                const_varray_view<T> view = front(dim);
                view.data_ += (len_[dim]-1)*stride_[dim];
                return view;
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                                !detail::are_convertible<idx_type, Args...>::value,
                                const_varray_view<T>>
            operator()(Args&&...) const
            {
                //TODO
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_convertible<idx_type, Args...>::value,
                                const_reference>
            operator()(Args&&...) const
            {
                //TODO
            }

            const_pointer data() const
            {
                return data_;
            }

            const_pointer data(const_pointer ptr)
            {
                std::swap(const_cast<pointer&>(ptr), data_);
                return ptr;
            }

            idx_type length(unsigned dim) const
            {
                assert(dim < dimension());
                return len_[dim];
            }

            idx_type length(unsigned dim, idx_type len)
            {
                assert(dim < dimension());
                std::swap(len, len_[dim]);
                return len;
            }

            const std::vector<idx_type>& lengths() const
            {
                return len_;
            }

            stride_type stride(unsigned dim) const
            {
                assert(dim < dimension());
                return stride_[dim];
            }

            stride_type stride(unsigned dim, stride_type stride)
            {
                assert(dim < dimension());
                std::swap(stride, stride_[dim]);
                return stride;
            }

            const std::vector<stride_type>& strides() const
            {
                return stride_;
            }

            unsigned dimension() const
            {
                return static_cast<unsigned>(len_.size());
            }
            
            void swap(const_varray_view& other)
            {
                using std::swap;
                swap(data_,   other.data_);
                swap(len_,    other.len_);
                swap(stride_, other.stride_);
            }

            friend void swap(const_varray_view& a, const_varray_view& b)
            {
                a.swap(b);
            }
    };

    template <typename T>
    class varray_view : protected const_varray_view<T>
    {
        template <typename T_> friend class const_varray_view;
        template <typename T_> friend class varray_view;
        template <typename T_, typename Allocator_> friend class varray;

        protected:
            typedef const_varray_view<T> base;

        public:
            typedef typename base::idx_type idx_type;
            typedef typename base::size_type size_type;
            typedef typename base::stride_type stride_type;
            typedef typename base::value_type value_type;
            typedef typename base::pointer pointer;
            typedef typename base::const_pointer const_pointer;
            typedef typename base::reference reference;
            typedef typename base::const_reference const_reference;

        protected:
            using base::data_;
            using base::len_;
            using base::stride_;

        public:
            using base::default_strides;

            varray_view() {}

            varray_view(const varray_view& other)
            : base(other) {}

            template <typename Alloc, typename=
                detail::enable_if_not_integral_t<Alloc>>
            varray_view(const varray<T, Alloc>& other)
            : base(other) {}

            varray_view(varray_view&& other)
            : base(std::move(other)) {}

            varray_view(const std::vector<idx_type>& len, pointer ptr, Layout layout=DEFAULT)
            {
                reset(len, ptr, layout);
            }

            template <typename U, typename=
                detail::enable_if_integral_t<U>>
            varray_view(const std::vector<U>& len, pointer ptr, Layout layout=DEFAULT)
            {
                reset(len, ptr, layout);
            }

            varray_view(const std::vector<idx_type>& len, pointer ptr, const std::vector<stride_type>& stride)
            {
                reset(len, ptr, stride);
            }

            template <typename U, typename V, typename=
                detail::enable_if_t<std::is_integral<U>::value &&
                                     std::is_integral<V>::value>>
            varray_view(const std::vector<U>& len, pointer ptr, const std::vector<V>& stride)
            {
                reset(len, ptr, stride);
            }

            void reset()
            {
                base::reset();
            }

            void reset(const varray_view<T>& other)
            {
                base::reset(other);
            }

            template <typename Alloc>
            detail::enable_if_not_integral_t<Alloc>
            reset(const varray<T, Alloc>& other)
            {
                base::reset(other);
            }

            void reset(varray_view<T>&& other)
            {
                base::reset(std::move(other));
            }

            void reset(const std::vector<idx_type>& len, pointer ptr, Layout layout = DEFAULT)
            {
                base::reset(len, ptr, layout);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            reset(const std::vector<U>& len, pointer ptr, Layout layout = DEFAULT)
            {
                base::reset(len, ptr, layout);
            }

            void reset(const std::vector<idx_type>& len, pointer ptr, const std::vector<stride_type>& stride)
            {
                base::reset(len, ptr, stride);
            }

            template <typename U, typename V>
            detail::enable_if_t<std::is_integral<U>::value &&
                                 std::is_integral<V>::value>
            reset(const std::vector<U>& len, pointer ptr, const std::vector<V>& stride)
            {
                base::reset(len, ptr, stride);
            }

            const varray_view& operator=(const const_varray_view<T>& other) const
            {
                copy(other, *this);
                return *this;
            }

            const varray_view& operator=(const varray_view<T>& other) const
            {
                copy(other, *this);
                return *this;
            }

            template <typename Alloc>
            const varray_view& operator=(const varray<T, Alloc>& other) const
            {
                copy(other, *this);
                return *this;
            }

            const varray_view& operator=(const T& value) const
            {
                auto it = make_iterator(len_, stride_);
                auto a_ = data_;
                while (it.next(a_)) *a_ = value;
                return *this;
            }

            using base::shift;
            using base::shift_down;
            using base::shift_up;

            varray_view<T> shifted(unsigned dim, idx_type n) const
            {
                return base::shifted(dim, n);
            }

            varray_view<T> shifted_down(unsigned dim) const
            {
                return base::shifted_down(dim);
            }

            varray_view<T> shifted_up(unsigned dim) const
            {
                return base::shifted_up(dim);
            }

            using base::permute;

            varray_view<T> permuted(const std::vector<unsigned>& perm) const
            {
                return base::permuted(perm);
            }

            template <typename U>
            detail::enable_if_integral_t<U,varray_view<T>>
            permuted(const std::vector<U>& perm) const
            {
                return base::permuted(perm);
            }

            using base::lower;

            varray_view<T> lowered(const std::vector<unsigned>& split) const
            {
                return base::lowered(split);
            }

            template <typename U>
            detail::enable_if_integral_t<U,varray_view<T>>
            lowered(const std::vector<U>& split) const
            {
                return base::lowered(split);
            }

            void rotate_dim(unsigned dim, idx_type shift)
            {
                assert(dim < dimension());

                idx_type n = len_[dim];
                stride_type s = stride_[dim];

                shift = shift%n;
                if (shift < 0) shift += n;

                if (shift == 0) return;

                std::vector<idx_type> sublen(dimension()-1);
                std::vector<stride_type> substride(dimension()-1);

                std::copy_n(len_.begin(), dim, sublen.begin());
                std::copy_n(len_.begin()+dim+1, dimension()-dim-1, sublen.begin()+dim);

                std::copy_n(stride_.begin(), dim, substride.begin());
                std::copy_n(stride_.begin()+dim+1, dimension()-dim-1, substride.begin()+dim);

                pointer p = data_;
                auto it = make_iterator(sublen, substride);
                while (it.next(p))
                {
                    pointer a = p;
                    pointer b = p+(shift-1)*s;
                    for (idx_type i = 0;i < shift/2;i++)
                    {
                        std::swap(*a, *b);
                        a += s;
                        b -= s;
                    }

                    a = p+shift*s;
                    b = p+(n-1)*s;
                    for (idx_type i = 0;i < (n-shift)/2;i++)
                    {
                        std::swap(*a, *b);
                        a += s;
                        b -= s;
                    }

                    a = p;
                    b = p+(n-1)*s;
                    for (idx_type i = 0;i < n/2;i++)
                    {
                        std::swap(*a, *b);
                        a += s;
                        b -= s;
                    }
                }
            }

            void rotate(const std::vector<idx_type>& shift)
            {
                rotate<idx_type>(shift);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            rotate(const std::vector<U>& shift)
            {
                assert(shift.size() == dimension());
                for (unsigned dim = 0;dim < dimension();dim++)
                {
                    rotate_dim(dim, shift[dim]);
                }
            }

            reference front() const
            {
                return const_cast<reference>(base::front());
            }

            varray_view<T> front(unsigned dim) const
            {
                return base::front(dim);
            }

            reference back() const
            {
                return const_cast<reference>(base::back());
            }

            varray_view<T> back(unsigned dim) const
            {
                return base::back(dim);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                                !detail::are_convertible<idx_type, Args...>::value,
                                varray_view<T>>
            operator()(Args&&... args) const
            {
                return base::operator()(std::forward<Args>(args)...);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_convertible<idx_type, Args...>::value,
                                reference>
            operator()(Args&&... args) const
            {
                return const_cast<reference>(base::operator()(std::forward<Args>(args)...));
            }

            pointer data() const
            {
                return const_cast<pointer>(base::data());
            }

            pointer data(pointer ptr)
            {
                return const_cast<pointer>(base::data(ptr));
            }

            using base::length;
            using base::lengths;
            using base::stride;
            using base::strides;
            using base::dimension;

            void swap(varray_view& other)
            {
                base::swap(other);
            }

            friend void swap(varray_view& a, varray_view& b)
            {
                a.swap(b);
            }
    };

    template <typename T, typename Allocator=std::allocator<T>>
    class varray : protected varray_view<T>, private Allocator
    {
        template <typename T_> friend class const_varray_view;
        template <typename T_> friend class varray_view;
        template <typename T_, typename Allocator_> friend class varray;

        protected:
            typedef varray_view<T> base;

        public:
            typedef typename base::idx_type idx_type;
            typedef typename base::size_type size_type;
            typedef typename base::stride_type stride_type;
            typedef typename base::value_type value_type;
            typedef typename base::pointer pointer;
            typedef typename base::const_pointer const_pointer;
            typedef typename base::reference reference;
            typedef typename base::const_reference const_reference;

        protected:
            using base::data_;
            using base::len_;
            using base::stride_;
            size_t size_ = 0;
            Layout layout_ = DEFAULT;

        public:
            using base::default_strides;

            varray() {}

            varray(const const_varray_view<T>& other, Layout layout=DEFAULT)
            {
                reset(other, layout);
            }

            varray(const varray_view<T>& other, Layout layout=DEFAULT)
            {
                reset(other, layout);
            }

            template <typename OAlloc, typename=
                detail::enable_if_not_integral_t<OAlloc>>
            varray(const varray<T, OAlloc>& other, Layout layout=DEFAULT)
            {
                reset(other, layout);
            }

            varray(const varray& other)
            {
                reset(other);
            }

            varray(varray&& other)
            {
                reset(std::move(other));
            }

            explicit varray(const std::vector<idx_type>& len, const T& val=T(), Layout layout=DEFAULT)
            {
                reset(len, val, layout);
            }

            template <typename U, typename=
                detail::enable_if_integral_t<U>>
            explicit varray(const std::vector<U>& len, const T& val=T(), Layout layout=DEFAULT)
            {
                reset(len, val, layout);
            }

            varray(const std::vector<idx_type>& len, uninitialized_t u, Layout layout=DEFAULT)
            {
                reset(len, u, layout);
            }

            template <typename U, typename=
                detail::enable_if_integral_t<U>>
            varray(const std::vector<U>& len, uninitialized_t u, Layout layout=DEFAULT)
            {
                reset(len, u, layout);
            }

            ~varray()
            {
                reset();
            }

            const varray& operator=(const const_varray_view<T>& other) const
            {
                base::operator=(other);
                return *this;
            }

            const varray& operator=(const varray_view<T>& other) const
            {
                base::operator=(other);
                return *this;
            }

            const varray& operator=(const varray& other) const
            {
                base::operator=(other);
                return *this;
            }

            template <typename OAlloc>
            const varray& operator=(const varray<T, OAlloc>& other) const
            {
                base::operator=(other);
                return *this;
            }

            const varray& operator=(const T& value) const
            {
                base::operator=(value);
                return *this;
            }

            void reset()
            {
                if (data_)
                {
                    for (size_t i = 0;i < size_;i++) data_[i].~T();
                    Allocator::deallocate(data_, size_);
                }
                size_ = 0;
                layout_ = DEFAULT;

                base::reset();
            }

            void reset(const const_varray_view<T>& other, Layout layout=DEFAULT)
            {
                if (std::is_scalar<T>::value)
                {
                    reset(other.len_, uninitialized, layout);
                }
                else
                {
                    reset(other.len_, T(), layout);
                }

                *this = other;
            }

            void reset(const varray_view<T>& other, Layout layout=DEFAULT)
            {
                reset(static_cast<const const_varray_view<T>&>(other), layout);
            }

            template <typename OAlloc>
            detail::enable_if_not_integral_t<OAlloc>
            reset(const varray<T, OAlloc>& other, Layout layout=DEFAULT)
            {
                reset(static_cast<const const_varray_view<T>&>(other), layout);
            }

            void reset(varray&& other)
            {
                swap(other);
            }

            void reset(const std::vector<idx_type>& len, const T& val=T(), Layout layout=DEFAULT)
            {
                reset<idx_type>(len, val, layout);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            reset(const std::vector<U>& len, const T& val=T(), Layout layout=DEFAULT)
            {
                reset(len, uninitialized, layout);
                std::uninitialized_fill_n(data_, size_, val);
            }

            void reset(const std::vector<idx_type>& len, uninitialized_t, Layout layout=DEFAULT)
            {
                reset<idx_type>(len, uninitialized, layout);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            reset(const std::vector<U>& len, uninitialized_t, Layout layout=DEFAULT)
            {
                size_ = std::accumulate(len.begin(), len.end(), size_t(1), std::multiplies<size_t>());
                layout_ = layout;

                base::reset(len, Allocator::allocate(size_), default_strides(len, layout));
            }

            void resize(const std::vector<idx_type>& len, const T& val=T())
            {
                resize<idx_type>(len, val);
            }

            template <typename U>
            detail::enable_if_integral_t<U>
            resize(const std::vector<U>& len, const T& val=T())
            {
                varray a(std::move(*this));
                reset(len, val, layout_);
                varray_view<T> b(*this);

                /*
                 * It is OK to change the geometry of 'a' even if it is not
                 * a view since it is about to go out of scope.
                 */
                for (unsigned i = 0;i < dimension();i++)
                {
                    a.len_[i] = b.len_[i] = std::min(a.len_[i], b.len_[i]);
                }

                copy(a, b);
            }

            void push_back(const T& x)
            {
                assert(base::ndim_ == 1);
                resize(base::len_[0]+1);
                back() = x;
            }

            void push_back(unsigned dim, const varray_view<T>& x)
            {
                assert(x.ndim_+1 == dimension());
                assert(dim < dimension());

                for (unsigned i = 0, j = 0;i < dimension();i++)
                {
                    assert(i == dim || len_[i] == x.len_[j++]);
                }

                std::vector<idx_type> len = len_;
                len[dim]++;
                resize(len);
                back(dim) = x;
            }

            void pop_back()
            {
                assert(base::ndim_ == 1);
                resize(base::len_[0]-1);
            }

            void pop_back(unsigned dim)
            {
                assert(dim < dimension());
                assert(base::len_[dim] > 0);

                std::vector<idx_type> len = len_;
                len[dim]--;
                resize(len);
            }

            using base::permute;

            varray_view<T> permuted(const std::vector<unsigned>& perm)
            {
                return base::permuted(perm);
            }

            template <typename U>
            detail::enable_if_integral_t<U,varray_view<T>>
            permuted(const std::vector<U>& perm)
            {
                return base::permuted(perm);
            }

            const_varray_view<T> permuted(const std::vector<unsigned>& perm) const
            {
                return base::permuted(perm);
            }

            template <typename U>
            detail::enable_if_integral_t<U,const_varray_view<T>>
            permuted(const std::vector<U>& perm) const
            {
                return base::permuted(perm);
            }

            using base::lower;

            varray_view<T> lowered(const std::vector<unsigned>& split)
            {
                return base::lowered(split);
            }

            template <typename U>
            detail::enable_if_integral_t<U,varray_view<T>>
            lowered(const std::vector<U>& split)
            {
                return base::lowered(split);
            }

            const_varray_view<T> lowered(const std::vector<unsigned>& split) const
            {
                return base::lowered(split);
            }

            template <typename U>
            detail::enable_if_integral_t<U,const_varray_view<T>>
            lowered(const std::vector<U>& split) const
            {
                return base::lowered(split);
            }

            using base::rotate_dim;
            using base::rotate;

            reference front()
            {
                return base::front();
            }

            const_reference front() const
            {
                return base::front();
            }

            varray_view<T> front(unsigned dim)
            {
                return base::front(dim);
            }

            const_varray_view<T> front(unsigned dim) const
            {
                return base::front(dim);
            }

            reference back()
            {
                return base::back();
            }

            const_reference back() const
            {
                return base::back();
            }

            varray_view<T> back(unsigned dim)
            {
                return base::back(dim);
            }

            const_varray_view<T> back(unsigned dim) const
            {
                return base::back(dim);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                                !detail::are_convertible<idx_type, Args...>::value,
                                varray_view<T>>
            operator()(Args&&... args)
            {
                return base::operator()(std::forward<Args>(args)...);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                                !detail::are_convertible<idx_type, Args...>::value,
                                const_varray_view<T>>
            operator()(Args&&... args) const
            {
                return base::operator()(std::forward<Args>(args)...);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_convertible<idx_type, Args...>::value,
                                reference>
            operator()(Args&&... args)
            {
                return base::operator()(std::forward<Args>(args)...);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_convertible<idx_type, Args...>::value,
                                const_reference>
            operator()(Args&&... args) const
            {
                return base::operator()(std::forward<Args>(args)...);
            }

            pointer data()
            {
                return base::data();
            }

            const_pointer data() const
            {
                return base::data();
            }

            idx_type length(unsigned dim) const
            {
                return base::length(dim);
            }

            const std::vector<idx_type>& lengths() const
            {
                return base::lengths();
            }

            stride_type stride(unsigned dim) const
            {
                return base::stride(dim);
            }

            const std::vector<stride_type>& strides() const
            {
                return base::strides();
            }

            unsigned dimension() const
            {
                return base::dimension();
            }

            void swap(varray& other)
            {
                using std::swap;
                base::swap(other);
                swap(size_,   other.size_);
                swap(layout_, other.layout_);
            }

            friend void swap(varray& a, varray& b)
            {
                a.swap(b);
            }
    };

    template <typename T>
    void copy(const_varray_view<T> a, varray_view<T> b)
    {
        assert(a.lengths() == b.lengths());

        auto it = make_iterator(a.lengths(), a.strides(), b.strides());
        auto a_ = a.data();
        auto b_ = b.data();
        while (it.next(a_, b_)) *b_ = *a_;
    }

}

#endif
