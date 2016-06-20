#ifndef _STL_EXT_BOUNDED_VECTOR_HPP_
#define _STL_EXT_BOUNDED_VECTOR_HPP_

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>

#include "fill_iterator.hpp"
#include "type_traits.hpp"

namespace stl_ext
{

template <typename T, size_t N, typename Allocator = std::allocator<T>>
class bounded_vector
{
    private:
        typedef std::allocator_traits<Allocator> alloc_traits_;

    public:
        typedef T value_type;
        typedef Allocator allocator_type;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename alloc_traits_::pointer pointer;
        typedef typename alloc_traits_::const_pointer const_pointer;
        typedef T* iterator;
        typedef const T* const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        explicit bounded_vector(const Allocator& alloc = Allocator())
        : allocator_(alloc) {}

        bounded_vector(size_type count,
                       const T& value,
                       const Allocator& alloc = Allocator())
        : allocator_(alloc)
        {
            assign(count, value);
        }

        explicit bounded_vector(size_type count)
        {
            resize(count);
        }

        template <typename InputIterator>
        bounded_vector(InputIterator first, InputIterator last,
                       const Allocator& alloc = Allocator())
        : allocator_(alloc)
        {
            assign(first, last);
        }

        bounded_vector(const bounded_vector& other)
        : allocator_(other.allocator_)
        {
            for (const T& x : other)
            {
                push_back(x);
            }
        }

        bounded_vector(bounded_vector&& other)
        : allocator_(other.allocator_)
        {
            for (T& x : other)
            {
                push_back(std::move(x));
            }
        }

        bounded_vector(const bounded_vector& other, const Allocator& alloc)
        : allocator_(alloc)
        {
            assign(other.begin(), other.end());
        }

        bounded_vector(std::initializer_list<T> init,
                       const Allocator& alloc = Allocator())
        : allocator_(alloc)
        {
            assign(init);
        }

        ~bounded_vector()
        {
            clear();
        }

        bounded_vector& operator=(const bounded_vector& other)
        {
            if (this == &other) return *this;

            clear();

            if (std::allocator_traits<allocator_type>::
                    propagate_on_container_copy_assignment::value)
                allocator_ = other.allocator_;

            for (const T& x : other) push_back(x);

            return *this;
        }

        bounded_vector& operator=(bounded_vector&& other)
        {
            if (this == &other) return *this;

            clear();

            if (std::allocator_traits<allocator_type>::
                    propagate_on_container_copy_assignment::value)
                allocator_ = other.allocator_;

            for (T& x : other) push_back(std::move(x));

            return *this;
        }

        bounded_vector& operator=(std::initializer_list<T> ilist)
        {
            assign(ilist);
            return *this;
        }

        void assign(size_type count, const T& value)
        {
            while (count --> 0) push_back(value);
        }

        template <typename InputIterator,
                  typename=enable_if_t<is_convertible<typename
                      std::iterator_traits<InputIterator>::iterator_category,
                      std::input_iterator_tag>::value>>
        void assign(InputIterator first, InputIterator last)
        {
            while (first != last)
            {
                push_back(*first);
                ++first;
            }
        }

        void assign(std::initializer_list<T> ilist)
        {
            clear();
            for (const T& x : ilist) push_back(x);
        }

        allocator_type get_allocator() const
        {
            return allocator_;
        }

        reference at(size_type pos)
        {
            if (pos >= size_) throw std::out_of_range("index out of range");
            return data()[pos];
        }

        const_reference at(size_type pos) const
        {
            if (pos >= size_) throw std::out_of_range("index out of range");
            return data()[pos];
        }

        reference operator[](size_type pos)
        {
            return data()[pos];
        }

        const_reference operator[](size_type pos) const
        {
            return data()[pos];
        }

        reference front()
        {
            return data()[0];
        }

        const_reference front() const
        {
            return data()[0];
        }

        reference back()
        {
            return data()[size_-1];
        }

        const_reference back() const
        {
            return data()[size_-1];
        }

        T* data()
        {
            return reinterpret_cast<T*>(allocator_.data_);
        }

        const T* data() const
        {
            return reinterpret_cast<const T*>(allocator_.data_);
        }

        iterator begin()
        {
            return data();
        }

        const_iterator begin() const
        {
            return data();
        }

        const_iterator cbegin() const
        {
            return data();
        }

        iterator end()
        {
            return data()+size_;
        }

        const_iterator end() const
        {
            return data()+size_;
        }

        const_iterator cend() const
        {
            return data()+size_;
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator(end());
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator(end());
        }

        reverse_iterator rend()
        {
            return reverse_iterator(begin());
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator(begin());
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator(begin());
        }

        bool empty() const
        {
            return size_ == 0;
        }

        size_type size() const
        {
            return size_;
        }

        size_type max_size() const
        {
            return N;
        }

        size_type capacity() const
        {
            return N;
        }

        void clear()
        {
            erase(begin(), end());
        }

        iterator insert(const_iterator pos_, const T& value)
        {
            iterator pos = begin()+(pos_-cbegin());

            alloc_traits_::construct(allocator_, data()+size_,
                                     std::move(back()));
            ++size_;

            std::move_backward(begin(), end()-1, begin()+1);
            *pos = value;

            return pos;
        }

        iterator insert(const_iterator pos_, T&& value)
        {
            iterator pos = begin()+(pos_-cbegin());

            alloc_traits_::construct(allocator_, data()+size_,
                                     std::move(back()));
            ++size_;

            std::move_backward(pos, end()-1, pos+1);
            *pos = std::move(value);

            return pos;
        }

        iterator insert(const_iterator pos, size_type count, const T& value)
        {
            return insert(pos, fill_begin(count, value), fill_end(count, value));
        }

        template <typename InputIterator,
                  typename=enable_if_t<is_convertible<typename
                      std::iterator_traits<InputIterator>::iterator_category,
                      std::input_iterator_tag>::value>>
        iterator insert(const_iterator pos_, InputIterator first,
                        InputIterator last)
        {
            iterator pos = begin()+(pos_-cbegin());
            insert_(pos, first, last,
                    typename std::iterator_traits<InputIterator>::
                        iterator_category());
            return pos;
        }

        iterator insert(const_iterator pos, std::initializer_list<T> ilist)
        {
            return insert(pos, ilist.begin(), ilist.end());
        }

        template <typename... Args>
        iterator emplace(const_iterator pos_, Args&&... args)
        {
            iterator pos = begin()+(pos_-cbegin());

            alloc_traits_::construct(allocator_, data()+size_,
                                     std::move(back()));
            ++size_;

            std::move_backward(pos, end()-1, pos+1);
            *pos = T(std::forward<Args>(args)...);

            return pos;
        }

        iterator erase(const_iterator pos_)
        {
            iterator pos = begin()+(pos_-cbegin());

            std::move(pos+1, end(), pos);
            --size_;

            alloc_traits_::destroy(allocator_, data()+size_);

            return pos;
        }

        iterator erase(const_iterator first_, const_iterator last_)
        {
            iterator first = begin()+(first_-cbegin());
            iterator last = begin()+(last_-cbegin());

            std::move(last, end(), first);

            while (first != last)
            {
                --size_;
                --last;
                alloc_traits_::destroy(allocator_, data()+size_);
            }

            return first;
        }

        void push_back(const T& value)
        {
            alloc_traits_::construct(allocator_, data()+size_, value);
            ++size_;
        }

        void push_back(T&& value)
        {
            alloc_traits_::construct(allocator_, data()+size_,
                                     std::move(value));
            ++size_;
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            alloc_traits_::construct(allocator_, data()+size_,
                                     std::forward<Args>(args)...);
            ++size_;
        }

        void pop_back()
        {
            --size_;
            alloc_traits_::destroy(allocator_, data()+size_);
        }

        void resize(size_type count)
        {
            while (size_ < count)
            {
                alloc_traits_::construct(allocator_, data()+size_);
                ++size_;
            }

            while (size_ > count)
            {
                --size_;
                alloc_traits_::destroy(allocator_, data()+size_);
            }
        }

        void resize(size_type count, const value_type& value)
        {
            while (size_ < count)
            {
                alloc_traits_::construct(allocator_, data()+size_, value);
                ++size_;
            }

            while (size_ > count)
            {
                --size_;
                alloc_traits_::destroy(allocator_, data()+size_);
            }
        }

        void swap(bounded_vector& other)
        {
            if (this == &other) return;

            if (size_ < other.size_)
            {
                std::swap_ranges(begin(), end(), other.begin());

                size_type old_size = size_;
                for (auto i = other.begin()+old_size;i < other.end();++i)
                {
                    push_back(std::move(*i));
                }

                other.resize(old_size);
            }
            else
            {
                std::swap_ranges(other.begin(), other.end(), begin());

                size_type old_size = other.size_;
                for (auto i = begin()+old_size;i < end();++i)
                {
                    other.push_back(std::move(*i));
                }

                resize(old_size);
            }
        }

    private:

        template <typename InputIterator>
        void insert_(iterator pos, InputIterator first,
                     InputIterator last, std::input_iterator_tag tag)
        {
            while (first != last)
            {
                insert(pos, *first);
                ++first;
                ++pos;
            }
        }

        template <typename ForwardIterator>
        void insert_(iterator pos, ForwardIterator first,
                     ForwardIterator last, std::forward_iterator_tag tag)
        {
            difference_type count = std::distance(first, last);
            difference_type tail = std::min(end()-pos, count);
            iterator end_ = end();

            construct_(end_+count-tail,
                      std::make_move_iterator(end_-tail),
                      std::make_move_iterator(end_));

            std::move_backward(pos, end_-tail, pos+tail);

            for (difference_type i = 0;i < tail;i++)
            {
                *pos = *first;
                ++pos;
                ++first;
            }

            construct_(end_, first, last);
        }

        template <typename InputIterator>
        void construct_(iterator pos, InputIterator first, InputIterator last)
        {
            while (first != last)
            {
                alloc_traits_::construct(allocator_, pos, *first);
                ++size_;
                ++pos;
                ++first;
            }
        }

        struct allocator_data : Allocator
        {
            allocator_data() {}

            allocator_data(const Allocator& alloc)
            : Allocator(alloc) {}

            allocator_data& operator=(const Allocator& alloc)
            {
                Allocator::operator=(alloc);
                return *this;
            }

            alignas(T) char data_[sizeof(T)*N];
        } allocator_;

        size_type size_ = 0;
};

template <typename T, size_t N, typename Allocator>
void swap(bounded_vector<T,N,Allocator>& lhs,
          bounded_vector<T,N,Allocator>& rhs)
{
    lhs.swap(rhs);
}

template <typename T, size_t N, typename Allocator>
bool operator==(const bounded_vector<T,N,Allocator>& lhs,
                const bounded_vector<T,N,Allocator>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T, size_t N, typename Allocator>
bool operator!=(const bounded_vector<T,N,Allocator>& lhs,
                const bounded_vector<T,N,Allocator>& rhs)
{
    return !(lhs == rhs);
}

template <typename T, size_t N, typename Allocator>
bool operator<(const bounded_vector<T,N,Allocator>& lhs,
                const bounded_vector<T,N,Allocator>& rhs)
{
    return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                        rhs.begin(), rhs.end());
}

template <typename T, size_t N, typename Allocator>
bool operator>(const bounded_vector<T,N,Allocator>& lhs,
                const bounded_vector<T,N,Allocator>& rhs)
{
    return rhs < lhs;
}

template <typename T, size_t N, typename Allocator>
bool operator<=(const bounded_vector<T,N,Allocator>& lhs,
                const bounded_vector<T,N,Allocator>& rhs)
{
    return !(rhs < lhs);
}

template <typename T, size_t N, typename Allocator>
bool operator>=(const bounded_vector<T,N,Allocator>& lhs,
                const bounded_vector<T,N,Allocator>& rhs)
{
    return !(lhs < rhs);
}

}

#endif
