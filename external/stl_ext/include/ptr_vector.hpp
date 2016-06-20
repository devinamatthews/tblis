#ifndef _STL_EXT_PTR_VECTOR_HPP_
#define _STL_EXT_PTR_VECTOR_HPP_

#include <initializer_list>
#include <iterator>
#include <memory>
#include <vector>

#include "global_ptr.hpp"
#include "type_traits.hpp"

namespace stl_ext
{

namespace detail
{

template <typename vector_>
class ptr_vector_
{
    private:
        template <typename ptr_>
        class iterator_
        {
            friend class ptr_vector_;

            public:
                typedef typename std::random_access_iterator_tag iterator_category;
                typedef typename std::iterator_traits<ptr_>::value_type value_type;
                typedef typename std::iterator_traits<ptr_>::difference_type difference_type;
                typedef typename std::iterator_traits<ptr_>::pointer pointer;
                typedef const remove_pointer_t<pointer> * const_pointer;
                typedef typename std::iterator_traits<ptr_>::reference reference;
                typedef const remove_reference_t<reference> & const_reference;

                typedef conditional_t<is_same<ptr_,value_type*>::value,
                                      typename vector_::iterator,
                                      typename vector_::const_iterator> ptr_iterator;

                iterator_() = default;

                iterator_(const iterator_& other)
                : it_(other.it_) {}

                template <typename T=pointer, typename=enable_if_const_pointer_t<T>>
                iterator_(const iterator_<value_type*>& other)
                : it_(other.it_) {}

                iterator_& operator=(const iterator_& other)
                {
                    it_ = other.it_;
                    return *this;
                }

                template <typename T=pointer, typename=enable_if_const_pointer_t<T>>
                iterator_& operator=(const iterator_<value_type*>& other)
                {
                    it_ = other.it_;
                    return *this;
                }

                bool operator==(const iterator_<value_type*>& x) const
                {
                    return it_ == x.it_;
                }

                bool operator!=(const iterator_<value_type*>& x) const
                {
                    return it_ != x.it_;
                }

                bool operator==(const iterator_<const value_type*>& x) const
                {
                    return it_ == x.it_;
                }

                bool operator!=(const iterator_<const value_type*>& x) const
                {
                    return it_ != x.it_;
                }

                template <typename T=reference,
                          typename=enable_if_not_same_t<T,const_reference>>
                reference operator*()
                {
                    return **it_;
                }

                const_reference operator*() const
                {
                    return **it_;
                }

                template <typename T=pointer,
                          typename=enable_if_not_same_t<T,const_pointer>>
                pointer operator->()
                {
                    return &**it_;
                }

                const_pointer operator->() const
                {
                    return &**it_;
                }

                iterator_& operator++()
                {
                    ++it_;
                    return *this;
                }

                iterator_& operator--()
                {
                    --it_;
                    return *this;
                }

                iterator_ operator++(int x)
                {
                    return iterator_(it_++);
                }

                iterator_ operator--(int x)
                {
                    return iterator_(it_--);
                }

                iterator_ operator+(difference_type n) const
                {
                    return iterator_(it_+n);
                }

                iterator_ operator-(difference_type n) const
                {
                    return iterator_(it_-n);
                }

                friend iterator_ operator+(difference_type n, const iterator_& x)
                {
                    return x+n;
                }

                difference_type operator-(const iterator_<value_type*>& other) const
                {
                    return it_-other.it_;
                }

                difference_type operator-(const iterator_<const value_type*>& other) const
                {
                    return it_-other.it_;
                }

                bool operator<(const iterator_<value_type*>& x) const
                {
                    return it_ < x.it_;
                }

                bool operator>(const iterator_<value_type*>& x) const
                {
                    return it_ > x.it_;
                }

                bool operator<=(const iterator_<value_type*>& x) const
                {
                    return it_ <= x.it_;
                }

                bool operator>=(const iterator_<value_type*>& x) const
                {
                    return it_ >= x.it_;
                }

                bool operator<(const iterator_<const value_type*>& x) const
                {
                    return it_ < x.it_;
                }

                bool operator>(const iterator_<const value_type*>& x) const
                {
                    return it_ > x.it_;
                }

                bool operator<=(const iterator_<const value_type*>& x) const
                {
                    return it_ <= x.it_;
                }

                bool operator>=(const iterator_<const value_type*>& x) const
                {
                    return it_ >= x.it_;
                }

                iterator_& operator+=(difference_type n)
                {
                    it_ += n;
                    return *this;
                }

                iterator_& operator-=(difference_type n)
                {
                    it_ -= n;
                    return *this;
                }

                template <typename T=reference,
                          typename=enable_if_not_same_t<T,const_reference>>
                reference operator[](difference_type n)
                {
                    return *it_[n];
                }

                const_reference operator[](difference_type n) const
                {
                    return *it_[n];
                }

                friend void swap(iterator_& a, iterator_& b)
                {
                    using std::swap;
                    swap(a.it_, b.it_);
                }

                ptr_iterator& base()
                {
                    return it_;
                }

                const ptr_iterator& base() const
                {
                    return it_;
                }

            protected:
                iterator_(ptr_iterator it) : it_(it) {}

                ptr_iterator it_;
        };

    public:
        typedef typename vector_::value_type ptr_type;
        typedef decay_t<decltype(*std::declval<ptr_type>())> value_type;
        typedef const value_type* const_pointer;
        typedef conditional_t<is_same<ptr_type,const_pointer>::value,
                              const value_type*,
                              value_type*> pointer;
        typedef const value_type& const_reference;
        typedef conditional_t<is_same<ptr_type,const_pointer>::value,
                              const value_type&,
                              value_type&> reference;
        typedef iterator_<pointer> iterator;
        typedef iterator_<const_pointer> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
        typedef typename std::iterator_traits<iterator>::difference_type difference_type;
        typedef typename vector_::size_type size_type;

        typedef typename vector_::iterator ptr_iterator;
        typedef typename vector_::const_iterator const_ptr_iterator;
        typedef typename vector_::reverse_iterator reverse_ptr_iterator;
        typedef typename vector_::const_reverse_iterator const_reverse_ptr_iterator;

        ptr_vector_() {}

        explicit ptr_vector_(size_type n) : impl_(n) {}

        ptr_vector_(size_type n, const value_type& val)
        {
            assign(n, val);
        }

        ptr_vector_(size_type n, value_type&& val)
        {
            assign(n, std::forward<value_type>(val));
        }

        ptr_vector_(const ptr_vector_&) = default;

        ptr_vector_(ptr_vector_&&) = default;

        ptr_vector_(std::initializer_list<value_type> il)
        {
            assign(il);
        }

        ptr_vector_(std::initializer_list<pointer> il)
        {
            assign(il);
        }

        template <typename InputIterator, typename=
            enable_if_t<is_convertible<typename std::iterator_traits<InputIterator>::value_type,value_type>::value ||
                        is_convertible<typename std::iterator_traits<InputIterator>::value_type,ptr_type>::value>>
        ptr_vector_(InputIterator i0, InputIterator i1)
        {
            assign(i0, i1);
        }

        ptr_vector_& operator=(const ptr_vector_&) = default;

        ptr_vector_& operator=(ptr_vector_&&) = default;

        iterator begin()
        {
            return iterator(impl_.begin());
        }

        iterator end()
        {
            return iterator(impl_.end());
        }

        const_iterator begin() const
        {
            return const_iterator(impl_.begin());
        }

        const_iterator end() const
        {
            return const_iterator(impl_.end());
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator(impl_.end());
        }

        reverse_iterator rend()
        {
            return reverse_iterator(impl_.begin());
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator(impl_.end());
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator(impl_.begin());
        }

        const_iterator cbegin() const
        {
            return const_iterator(impl_.begin());
        }

        const_iterator cend() const
        {
            return const_iterator(impl_.end());
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator(impl_.cend());
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator(impl_.cbegin());
        }

        ptr_iterator pbegin()
        {
            return impl_.begin();
        }

        ptr_iterator pend()
        {
            return impl_.end();
        }

        const_ptr_iterator pbegin() const
        {
            return impl_.begin();
        }

        const_ptr_iterator pend() const
        {
            return impl_.end();
        }

        reverse_ptr_iterator rpbegin()
        {
            return impl_.rbegin();
        }

        reverse_ptr_iterator rpend()
        {
            return impl_.rend();
        }

        const_reverse_ptr_iterator rpbegin() const
        {
            return impl_.rbegin();
        }

        const_reverse_ptr_iterator rpend() const
        {
            return impl_.rend();
        }

        const_ptr_iterator cpbegin() const
        {
            return impl_.begin();
        }

        const_ptr_iterator cpend() const
        {
            return impl_.end();
        }

        const_reverse_ptr_iterator crpbegin() const
        {
            return impl_.rbegin();
        }

        const_reverse_ptr_iterator crpend() const
        {
            return impl_.rend();
        }

        size_type size() const
        {
            return impl_.size();
        }

        size_type max_size() const
        {
            return impl_.max_size();
        }

        void resize(size_type n)
        {
            impl_.resize(n);
        }

        void resize(size_type n, const value_type& x)
        {
            if (n <= impl_.size())
            {
                impl_.resize(n);
            }
            else
            {
                impl_.reserve(n);

                for (size_type i = impl_.size();i < n;i++)
                {
                    impl_.emplace_back(new value_type(x));
                }
            }
        }

        void resize(size_type n, value_type&& x)
        {
            if (n <= impl_.size())
            {
                impl_.resize(n);
            }
            else
            {
                impl_.reserve(n);

                impl_.emplace_back(new value_type(std::move(x)));

                for (size_type i = impl_.size();i < n;i++)
                {
                    impl_.emplace_back(new value_type(x));
                }
            }
        }

        size_type capacity() const
        {
            return impl_.capacity();
        }

        bool empty() const
        {
            return impl_.empty();
        }

        void reserve(size_type n)
        {
            impl_.reserve(n);
        }

        void shrink_to_fit()
        {
            impl_.shrink_to_fit();
        }

        reference operator[](size_type n)
        {
            return *impl_[n];
        }

        const_reference operator[](size_type n) const
        {
            return *impl_[n];
        }

        reference at(size_type n)
        {
            return *impl_.at(n);
        }

        const_reference at(size_type n) const
        {
            return *impl_.at(n);
        }

        ptr_type& ptr(size_type n)
        {
            return impl_.at(n);
        }

        const ptr_type& ptr(size_type n) const
        {
            return impl_.at(n);
        }

        reference front()
        {
            return *impl_.front();
        }

        reference back()
        {
            return *impl_.back();
        }

        const_reference front() const
        {
            return *impl_.front();
        }

        const_reference back() const
        {
            return *impl_.back();
        }

        ptr_type& pfront()
        {
            return impl_.front();
        }

        ptr_type& pback()
        {
            return impl_.back();
        }

        const ptr_type& pfront() const
        {
            return impl_.front();
        }

        const ptr_type& pback() const
        {
            return impl_.back();
        }

        void assign(size_type n, const value_type& val)
        {
            impl_.clear();
            resize(n, val);
        }

        void assign(size_type n, value_type&& val)
        {
            impl_.clear();
            resize(n, std::move(val));
        }

        void assign(const ptr_vector_& x)
        {
            *this = x;
        }

        void assign(ptr_vector_&& x)
        {
            *this = std::move(x);
        }

        void assign(std::initializer_list<value_type> il)
        {
            impl_.clear();
            impl_.reserve(il.size());

            for (auto& val : il)
            {
                impl_.emplace_back(new value_type(val));
            }
        }

        void assign(std::initializer_list<pointer> il)
        {
            impl_.clear();
            impl_.reserve(il.size());

            for (auto& ptr : il)
            {
                impl_.emplace_back(ptr);
            }
        }

        template <typename InputIterator>
        enable_if_convertible_t<typename std::iterator_traits<InputIterator>::value_type,value_type>
        assign(InputIterator i0, InputIterator i1)
        {
            impl_.clear();

            while (i0 != i1)
            {
                impl_.emplace_back(new value_type(*i0));
                i0++;
            }
        }

        template <typename InputIterator>
        enable_if_convertible_t<typename std::iterator_traits<InputIterator>::value_type,ptr_type>
        assign(InputIterator i0, InputIterator i1)
        {
            impl_.clear();

            while (i0 != i1)
            {
                impl_.emplace_back(*i0);
                i0++;
            }
        }

        void push_back(const value_type& x)
        {
            impl_.emplace_back(new value_type(x));
        }

        void push_back(value_type&& x)
        {
            impl_.emplace_back(new value_type(std::move(x)));
        }

        void push_back(const ptr_type& x)
        {
            impl_.push_back(x);
        }

        void push_back(ptr_type&& x)
        {
            impl_.push_back(std::move(x));
        }

        template <typename T=pointer, typename=enable_if_not_same_t<T,ptr_type>>
        void push_back(pointer x)
        {
            impl_.emplace_back(x);
        }

        void pop_back()
        {
            impl_.pop_back();
        }

        iterator insert(const_iterator position, const value_type& val)
        {
            return impl_.emplace(position.it_, new value_type(val));
        }

        iterator insert(const_iterator position, value_type&& val)
        {
            return impl_.emplace(position.it_, new value_type(std::move(val)));
        }

        iterator insert(const_iterator position, const ptr_type& val)
        {
            return impl_.insert(position.it_, val);
        }

        iterator insert(const_iterator position, ptr_type&& val)
        {
            return impl_.insert(position.it_, std::move(val));
        }

        iterator insert(const_iterator position, pointer val)
        {
            return impl_.emplace(position.it_, val);
        }

        iterator insert(const_iterator position, size_type n, const value_type& val)
        {
            auto m = position-begin();
            reserve(size()+n);
            auto middle = impl_.end();
            resize(size()+n, val);
            rotate(impl_.begin()+m, middle, impl_.end());
            return begin()+m;
        }

        iterator insert(const_iterator position, size_type n, value_type&& val)
        {
            auto m = position-begin();
            reserve(size()+n);
            auto middle = impl_.end();
            resize(size()+n, val);
            rotate(impl_.begin()+m, middle, impl_.end());
            return begin()+m;
        }

        iterator insert(const_iterator position, std::initializer_list<value_type> il)
        {
            auto m = position-begin();
            reserve(size()+il.size());
            auto middle = impl_.end();

            for (auto& val : il)
            {
                impl_.emplace_back(new value_type(val));
            }

            rotate(impl_.begin()+m, middle, impl_.end());
            return begin()+m;
        }

        iterator insert(const_iterator position, std::initializer_list<pointer> il)
        {
            auto m = position-begin();
            reserve(size()+il.size());
            auto middle = impl_.end();

            for (auto& ptr : il)
            {
                impl_.emplace_back(ptr);
            }

            rotate(impl_.begin()+m, middle, impl_.end());
            return begin()+m;
        }

        iterator erase(const_iterator position)
        {
            return iterator(impl_.erase(position.it_));
        }

        iterator erase(const_iterator first, const_iterator last)
        {
            return iterator(impl_.erase(first.it_, last.it_));
        }

        ptr_iterator perase(const_ptr_iterator position)
        {
            /*
             * Work around stupid bug in libstdc++ and libc++
             */
            ptr_iterator pos = impl_.begin()+(position-impl_.cbegin());
            return impl_.erase(pos);
        }

        ptr_iterator perase(const_ptr_iterator first, const_ptr_iterator last)
        {
            /*
             * Work around stupid bug in libstdc++ and libc++
             */
            ptr_iterator f = impl_.begin()+(first-impl_.cbegin());
            ptr_iterator l = impl_.begin()+(last-impl_.cbegin());
            return impl_.erase(f, l);
        }

        void swap(ptr_vector_& x)
        {
            impl_.swap(x.impl_);
        }

        void clear()
        {
            impl_.clear();
        }

        template <typename... Args>
        iterator emplace(const_iterator position, Args&&... args)
        {
            return impl_.emplace(position.it_, new value_type(std::forward<Args>(args)...));
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            impl_.emplace_back(new value_type(std::forward<Args>(args)...));
        }

        friend bool operator==(const ptr_vector_& lhs, const ptr_vector_& rhs)
        {
            if (lhs.size() != rhs.size()) return false;
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }

        friend bool operator!=(const ptr_vector_& lhs, const ptr_vector_& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator<(const ptr_vector_& lhs, const ptr_vector_& rhs)
        {
            return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                                rhs.begin(), rhs.end());
        }

        friend bool operator>(const ptr_vector_& lhs, const ptr_vector_& rhs)
        {
            return rhs < lhs;
        }

        friend bool operator<=(const ptr_vector_& lhs, const ptr_vector_& rhs)
        {
            return !(rhs < lhs);
        }

        friend bool operator>=(const ptr_vector_& lhs, const ptr_vector_& rhs)
        {
            return !(lhs < rhs);
        }

        friend void swap(ptr_vector_&& a, ptr_vector_&& b)
        {
            a.swap(b);
        }

        friend void swap(ptr_vector_& a, ptr_vector_& b)
        {
            a.swap(b);
        }

    private:
        vector_ impl_;
};

}

template <typename T>
using ptr_vector = detail::ptr_vector_<std::vector<T*>>;

template <typename T>
using unique_vector = detail::ptr_vector_<std::vector<std::unique_ptr<T>>>;

template <typename T>
using shared_vector = detail::ptr_vector_<std::vector<std::shared_ptr<T>>>;

template <typename T>
using global_vector = detail::ptr_vector_<std::vector<global_ptr<T>>>;

}

#endif
