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

    template <typename T> void copy(const const_varray_view<T>& a, varray_view<T>& b);
    template <typename T> void copy(const const_varray_view<T>& a, varray_view<T>&& b);

    template <typename T>
    class const_varray_view
    {
        template <typename T_> friend class const_varray_view;
        template <typename T_> friend class varray_view;
        template <typename T_, typename Allocator_> friend class varray;

        public:
            typedef unsigned idx_type;
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
            unsigned ndim_ = 0;

            const_varray_view& operator=(const const_varray_view& other) = delete;

            void reset()
            {
                data_ = nullptr;
                len_.clear();
                stride_.clear();
                ndim_ = 0;
            }

            void reset(const varray_view<T>& other)
            {
                data_ = other.data_;
                len_ = other.len_;
                stride_ = other.stride_;
                ndim_ = other.ndim_;
            }

            void reset(const std::vector<idx_type>& len, const_pointer ptr, Layout layout = DEFAULT)
            {
                reset(len, ptr, default_strides(len, layout));
            }

            void reset(const std::vector<idx_type>& len, const_pointer ptr, const std::vector<stride_type>& stride)
            {
                assert(len.size() > 0);
                assert(len.size() == stride.size());
                data_ = const_cast<pointer>(ptr);
                len_ = len;
                stride_ = stride;
            }

        public:
            static std::vector<stride_type> default_strides(const std::vector<idx_type>& len, Layout layout=DEFAULT)
            {
                std::vector<stride_type> stride(len.size());

                int ndim = len.size();
                if (layout == ROW_MAJOR)
                {
                    stride[ndim-1] = 1;
                    for (unsigned i = ndim-1;i > 0;i--)
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

            const_varray_view(const const_varray_view& other)
            {
                reset(other);
            }

            const_varray_view(const std::vector<idx_type>& len, const_pointer ptr, Layout layout=DEFAULT)
            {
                reset(len, ptr, layout);
            }

            const_varray_view(const std::vector<idx_type>& len, const_pointer ptr, const std::vector<stride_type>& stride)
            {
                reset(len, ptr, stride);
            }

            const_varray_view<T> permute(const std::vector<unsigned>& perm) const
            {
                assert(perm.size() == ndim_);

                std::vector<idx_type> len(ndim_);
                std::vector<stride_type> stride(ndim_);

                for (unsigned i = 0;i < ndim_;i++)
                {
                    assert(0 <= perm[i] && perm[i] < ndim_);
                    for (unsigned j = 0;j < i;j++) assert(perm[i] != perm[j]);
                }

                for (unsigned i = 0;i < ndim_;i++)
                {
                    len[i] = len_[perm[i]];
                    stride[i] = stride_[perm[i]];
                }

                return {len, data_, stride};
            }

            const_varray_view<T> lower(const std::vector<unsigned>& split) const
            {
                assert(split.size() < ndim_);

                unsigned newdim = split.size();
                for (unsigned i = 0;i < newdim;i++)
                {
                    assert(split[i] <= ndim_);
                    if (i != 0) assert(split[i-1] <= split[i]);
                }

                std::vector<idx_type> newlen(newdim+1);
                std::vector<stride_type> newstride(newdim+1);

                for (unsigned i = 0;i <= newdim;i++)
                {
                    int begin = (i == 0 ? 0 : split[i-1]);
                    int end = (i == newdim-1 ? ndim_-1 : split[i]-1);
                    if (begin > end) continue;

                    if (stride_[begin] < stride_[end])
                    {
                        newlen[i] = len_[end];
                        newstride[i] = stride_[begin];
                        for (unsigned j = begin;j < end;j++)
                        {
                            assert(stride_[j+1] == stride_[j]*len_[j]);
                            newlen[i] *= len_[j];
                        }
                    }
                    else
                    {
                        newlen[i] = len_[end];
                        newstride[i] = stride_[end];
                        for (unsigned j = begin;j < end;j++)
                        {
                            assert(stride_[j] == stride_[j+1]*len_[j+1]);
                            newlen[i] *= len_[j];
                        }
                    }
                }

                return {newlen, data_, newstride};
            }

            const_reference front() const
            {
                assert(ndim_ == 1);
                assert(len_[0] > 0);
                return data_[0];
            }

            const_varray_view<T> front(unsigned dim) const
            {
                assert(ndim_ > 1);
                assert(dim < ndim_);
                assert(len_[dim] > 0);

                std::vector<idx_type> len(ndim_-1);
                std::vector<stride_type> stride(ndim_-1);

                std::copy_n(len_.begin(), dim, len.begin());
                std::copy_n(len_.begin()+dim+1, ndim_-dim-1, len.begin()+dim);
                std::copy_n(stride_.begin(), dim, stride.begin());
                std::copy_n(stride_.begin()+dim+1, ndim_-dim-1, stride.begin()+dim);

                return {len, data_, stride};
            }

            const_reference back() const
            {
                assert(ndim_ == 1);
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
            operator()(Args&&... args) const
            {
                //TODO
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_convertible<idx_type, Args...>::value,
                                const_reference>
            operator()(Args&&... args) const
            {
                //TODO
            }

            const_pointer data() const
            {
                return data_;
            }

            idx_type length() const
            {
                assert(ndim_ == 1);
                return len_[0];
            }

            idx_type length(unsigned dim) const
            {
                assert(dim < ndim_);
                return len_[dim];
            }

            const std::vector<idx_type>& lengths() const
            {
                return len_;
            }

            stride_type stride() const
            {
                assert(ndim_ == 1);
                return stride_[0];
            }

            stride_type stride(unsigned dim) const
            {
                assert(dim < ndim_);
                return stride_[dim];
            }

            const std::vector<stride_type>& strides() const
            {
                return stride_;
            }

            void swap(const_varray_view& other)
            {
                using std::swap;
                swap(data_,   other.data_);
                swap(len_,    other.len_);
                swap(stride_, other.stride_);
                swap(ndim_,   other.ndim_);
            }

            friend void swap(const_varray_view& other& a, const_varray_view& other& b)
            {
                a.swap(b);
            }
    };

    template <typename T>
    class varray_view : public const_varray_view<T>
    {
        template <typename T_> friend class const_varray_view;
        template <typename T_> friend class varray_view;
        template <typename T_, typename Allocator_> friend class varray;

        protected:
            typedef const_varray_view<T> base;
            typedef const_varray_view<T> parent;

        public:
            using typename base::idx_type;
            using typename base::size_type;
            using typename base::stride_type;
            using typename base::value_type;
            using typename base::pointer;
            using typename base::const_pointer;
            using typename base::reference;
            using typename base::const_reference;

        protected:
            using base::data_;
            using base::len_;
            using base::stride_;
            using base::ndim_;

            using base::reset;

            varray_view(const parent& other)
            : parent(other) {}

        public:
            varray_view() {}

            varray_view(varray_view& other)
            : parent(other) {}

            varray_view(varray_view&& other)
            : parent(other) {}

            varray_view(const std::vector<idx_type>& len, pointer ptr, Layout layout=DEFAULT)
            {
                reset(len, ptr, layout);
            }

            varray_view(const std::vector<idx_type>& len, pointer ptr, const std::vector<stride_type>& stride)
            {
                reset(len, ptr, stride);
            }

            varray_view& operator=(const const_varray_view<T>& other)
            {
                copy(other, *this);
                return *this;
            }

            varray_view& operator=(const varray_view& other)
            {
                return operator=(static_cast<const const_varray_view<T>&>(other));
            }

            varray_view& operator=(const T& value)
            {
                auto it = make_iterator(len_, stride_);
                auto a_ = data_;
                while (it.next(a_)) *a_ = value;
                return *this;
            }

            using base::permute;

            varray_view<T> permute(const std::vector<unsigned>& perm)
            {
                return base::permute(perm);
            }

            using base::lower;

            varray_view<T> lower(const std::vector<unsigned>& split)
            {
                return base::lower(split);
            }

            void rotate_dim(unsigned dim, stride_type shift)
            {
                assert(dim < ndim_);

                idx_type n = len_[dim];
                stride_type s = stride_[dim];

                shift = shift%n;
                if (shift < 0) shift += n;

                if (shift == 0) return;

                std::vector<idx_type> sublen(ndim_-1);
                std::vector<stride_type> substride(ndim_-1);

                std::copy_n(len_.begin(), dim, sublen.begin());
                std::copy_n(len_.begin()+dim+1, ndim-dim-1, sublen.begin()+dim);

                std::copy_n(stride_.begin(), dim, substride.begin());
                std::copy_n(stride_.begin()+dim+1, ndim-dim-1, substride.begin()+dim);

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

            void rotate(const std::vector<stride_type>& shift)
            {
                for (unsigned dim = 0;dim < ndim;dim++)
                {
                    rotate_dim(dim, shift[dim]);
                }
            }

            using base::front;

            reference front()
            {
                return const_cast<reference>(base::front());
            }

            varray_view<T> front(unsigned dim)
            {
                return base::front(dim);
            }

            using base::back;

            reference back()
            {
                return const_cast<reference>(base::back());
            }

            varray_view<T> back(unsigned dim)
            {
                return base::back(dim);
            }

            using base::operator();

            template <typename... Args>
            detail::enable_if_t<detail::are_indices_or_slices<Args...>::value &&
                                !detail::are_convertible<idx_type, Args...>::value,
                                varray_view<T>>
            operator()(Args&&... args)
            {
                return base::operator()(std::forward<Args>(args)...);
            }

            template <typename... Args>
            detail::enable_if_t<detail::are_convertible<idx_type, Args...>::value,
                                reference>
            operator()(Args&&... args)
            {
                return const_cast<reference>(base::operator()(std::forward<Args>(args)...));
            }

            using base::data;

            pointer data()
            {
                return const_cast<pointer>(base::data());
            }

            using base::swap;

            friend void swap(varray_view& a, varray_view& b)
            {
                a.swap(b);
            }
    };

    template <typename T, typename Allocator=aligned_allocator<T, MARRAY_BASE_ALIGNMENT>>
    class varray : public varray_view<T>, private Allocator
    {
        template <typename T_> friend class const_varray_view;
        template <typename T_> friend class varray_view;
        template <typename T_, typename Allocator_> friend class varray;

        protected:
            typedef const_varray_view<T> base;
            typedef varray_view<T> parent;

        public:
            using typename base::idx_type;
            using typename base::size_type;
            using typename base::stride_type;
            using typename base::value_type;
            using typename base::pointer;
            using typename base::const_pointer;
            using typename base::reference;
            using typename base::const_reference;

        protected:
            using base::data_;
            using base::len_;
            using base::stride_;
            using base::ndim_;
            size_t size_ = 0;
            Layout layout_ = DEFAULT;

        public:
            using base::default_strides;

            varray() {}

            varray(const varray_view<T>& other, Layout layout=DEFAULT)
            {
                reset(other);
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

            varray(const std::vector<idx_type>& len, uninitialized_t u, Layout layout=DEFAULT)
            {
                reset(len, u, layout);
            }

            ~varray()
            {
                reset();
            }

            using parent::operator=;

            varray& operator=(const varray& other)
            {
                parent::operator=(other);
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

            void reset(const varray_view<T>& other, Layout layout=DEFAULT)
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

            void reset(varray&& other)
            {
                swap(other);
            }

            void reset(const std::vector<idx_type>& len, const T& val=T(), Layout layout=DEFAULT)
            {
                reset(len, uninitialized, layout);
                std::uninitialized_fill_n(data_, size_, val);
            }

            void reset(const std::vector<idx_type>& len, uninitialized_t u, Layout layout=DEFAULT)
            {
                size_ = std::accumulate(len.begin(), len.end(), size_t(1), std::multiplies<size_t>());
                layout_ = layout;

                base::reset(len, Allocator::allocate(size_), base::default_strides(len, layout));
            }

            void resize(const std::vector<idx_type>& len, const T& val=T())
            {
                varray a(std::move(*this));
                reset(len, val, layout_);
                varray_view<T> b(*this);

                /*
                 * It is OK to change the geometry of 'a' even if it is not
                 * a view since it is about to go out of scope.
                 */
                for (unsigned i = 0;i < ndim;i++)
                {
                    a.len_[i] = b.len_[i] = std::min(a.len_[i], b.len_[i]);
                }

                copy(a, b);
            }

            void push_back(const T& x)
            {
                assert(ndim_ == 1);
                resize(len_[0]+1);
                back() = x;
            }

            void push_back(unsigned dim, const varray_view<T>& x)
            {
                assert(x.ndim_ == ndim_-1);
                assert(dim < ndim_);

                for (unsigned i = 0, j = 0;i < ndim;i++)
                {
                    if (i != dim)
                    {
                        assert(len_[i] == x.len_[j++]);
                    }
                }

                std::vector<idx_type> len = len_;
                len[dim]++;
                resize(len);
                back(dim) = x;
            }

            void pop_back()
            {
                assert(ndim_ == 1);
                resize(len_[0]-1);
            }

            void pop_back(unsigned dim)
            {
                assert(dim < ndim_);
                assert(len_[dim] > 0);

                std::vector<idx_type> len = len_;
                len[dim]--;
                resize(len);
            }

            using base::permute;
            using parent::permute;

            using base::lower;
            using parent::lower;

            using parent::rotate_dim;
            using parent::rotate;

            using base::front;
            using parent::front;

            using base::back;
            using parent::back;

            using base::operator();
            using parent::operator();

            using base::data;
            using parent::data;

            using base::length;
            using parent::length;

            using base::lengths;
            using parent::lengths;

            using base::stride;
            using parent::stride;

            using base::strides;
            using parent::strides;

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
    void copy(const const_varray_view<T>& a, varray_view<T>& b)
    {
        assert(a.lengths() == b.lengths());

        auto it = make_iterator(a.lengths(), a.strides(), b.strides());
        auto a_ = a.data();
        auto b_ = b.data();
        while (it.next(a_, b_)) *b_ = *a_;
    }

    template <typename T>
    void copy(const const_varray_view<T>& a, varray_view<T>&& b)
    {
        copy(a, b);
    }

}

#endif
