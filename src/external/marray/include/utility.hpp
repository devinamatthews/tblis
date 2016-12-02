#ifndef _MARRAY_UTILITY_HPP_
#define _MARRAY_UTILITY_HPP_

#include <type_traits>
#include <array>
#include <vector>
#include <utility>
#include <iterator>
#include <cassert>
#include <algorithm>

namespace MArray
{
    /*
     * Create a vector from the specified elements, where the type of the vector
     * is taken from the first element.
     */
    template <typename T, typename... Args>
    std::vector<typename std::decay<T>::type>
    make_vector(T&& t, Args&&... args)
    {
        return {{std::forward<T>(t), std::forward<Args>(args)...}};
    }

    /*
     * Create an array from the specified elements, where the type of the array
     * is taken from the first element.
     */
    template <typename T, typename... Args>
    std::array<typename std::decay<T>::type, sizeof...(Args)+1>
    make_array(T&& t, Args&&... args)
    {
        return {{std::forward<T>(t), std::forward<Args>(args)...}};
    }

    template <typename T>
    class range_t
    {
        static_assert(std::is_integral<T>::value, "The type must be integral.");

        protected:
            T from_;
            T to_;
            T delta_;

            typedef T value_type;
            typedef T size_type;

        public:
            class iterator : std::iterator<std::random_access_iterator_tag,T>
            {
                protected:
                    T val_;
                    T delta_;

                public:
                    using typename std::iterator<std::random_access_iterator_tag,T>::iterator_category;
                    using typename std::iterator<std::random_access_iterator_tag,T>::value_type;
                    using typename std::iterator<std::random_access_iterator_tag,T>::difference_type;
                    using typename std::iterator<std::random_access_iterator_tag,T>::pointer;
                    using typename std::iterator<std::random_access_iterator_tag,T>::reference;

                    constexpr iterator() : val_(0), delta_(0) {}

                    constexpr iterator(T val, T delta) : val_(val), delta_(delta) {}

                    bool operator==(const iterator& other)
                    {
                        return val_ == other.val_ && delta_ == other.delta_;
                    }

                    bool operator!=(const iterator& other)
                    {
                        return val_ != other.val_ || delta_ != other.delta_;
                    }

                    value_type operator*() const
                    {
                        return val_;
                    }

                    iterator& operator++()
                    {
                        val_ += delta_;
                        return *this;
                    }

                    iterator operator++(int)
                    {
                        iterator old(*this);
                        val_ += delta_;
                        return old;
                    }

                    iterator& operator--()
                    {
                        val_ -= delta_;
                        return *this;
                    }

                    iterator operator--(int)
                    {
                        iterator old(*this);
                        val_ -= delta_;
                        return old;
                    }

                    iterator& operator+=(difference_type n)
                    {
                        val_ += n*delta_;
                        return *this;
                    }

                    iterator operator+(difference_type n)
                    {
                        return iterator(val_+n*delta_);
                    }

                    friend iterator operator+(difference_type n, const iterator& i)
                    {
                        return iterator(i.val_+n*i.delta_);
                    }

                    iterator& operator-=(difference_type n)
                    {
                        val_ -= n*delta_;
                        return *this;
                    }

                    iterator operator-(difference_type n)
                    {
                        return iterator(val_-n*delta_);
                    }

                    difference_type operator-(const iterator& other)
                    {
                        return val_-other.val_;
                    }

                    bool operator<(const iterator& other)
                    {
                        return val_ < other.val_;
                    }

                    bool operator<=(const iterator& other)
                    {
                        return val_ <= other.val_;
                    }

                    bool operator>(const iterator& other)
                    {
                        return val_ > other.val_;
                    }

                    bool operator>=(const iterator& other)
                    {
                        return val_ >= other.val_;
                    }

                    value_type operator[](difference_type n) const
                    {
                        return val_+n*delta_;
                    }

                    friend void swap(iterator& a, iterator& b)
                    {
                        using std::swap;
                        swap(a.val_, b.val_);
                        swap(a.delta_, b.delta_);
                    }
            };

            constexpr range_t()
            : from_(0), to_(0), delta_(0) {}

            constexpr range_t(T from, T to, T delta)
            : from_(from), to_(from+((to-from+delta-1)/delta)*delta), delta_(delta) {}

            range_t(const range_t&) = default;

            range_t(range_t&&) = default;

            range_t& operator=(const range_t&) = default;

            range_t& operator=(range_t&&) = default;

            size_type size() const
            {
                return (to_-from_)/delta_;
            }

            iterator begin() const
            {
                return iterator(from_, delta_);
            }

            iterator end() const
            {
                return iterator(to_, delta_);
            }

            value_type front() const
            {
                return from_;
            }

            value_type back() const
            {
                return to_-delta_;
            }

            value_type operator[](size_type n) const
            {
                return from_+n*delta_;
            }

            operator std::vector<T>() const
            {
                return std::vector<T>(begin(), end());
            }

            template <typename T_=T, typename=typename std::enable_if<std::is_same<T_,char>::value>::type>
            operator std::string() const
            {
                return std::string(begin(), end());
            }
    };

    template <typename T>
    range_t<T> range(T to)
    {
        return {T(), to, 1};
    }

    template <typename T>
    range_t<T> range(T from, T to)
    {
        return {from, to, 1};
    }

    template <typename T>
    range_t<T> range(T from, T to, T delta)
    {
        return {from, to, delta};
    }

    template <typename T, size_t N, typename Allocator=std::allocator<T>>
    class short_vector
    {
        protected:
            typedef std::allocator_traits<Allocator> _alloc_traits;

        public:
            typedef T value_type;
            typedef Allocator allocator_type;
            typedef size_t size_type;
            typedef ptrdiff_t difference_type;
            typedef value_type& reference;
            typedef const value_type& const_reference;
            typedef typename _alloc_traits::pointer pointer;
            typedef typename _alloc_traits::const_pointer const_pointer;
            typedef pointer iterator;
            typedef const_pointer const_iterator;
            typedef std::reverse_iterator<iterator> reverse_iterator;
            typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

            explicit short_vector(const Allocator& alloc = Allocator())
            : _size(0), _alloc(alloc, _local_data()) {}

            short_vector(size_type count, const T& value,
                         const Allocator& alloc = Allocator())
            : _size(0), _alloc(alloc, _local_data())
            {
                assign(count, value);
            }

            explicit short_vector(size_type count)
            : _size(0), _alloc(Allocator(), _local_data())
            {
                assign(count, T());
            }

            template <typename Iterator>
            short_vector(Iterator first, Iterator last,
                         const Allocator& alloc = Allocator())
            : _size(0), _alloc(alloc, _local_data())
            {
                assign(first, last);
            }

            short_vector(const short_vector& other)
            : _size(0), _alloc(_alloc_traits::select_on_container_copy_construction(other._alloc),
                               _local_data())
            {
                assign(other.begin(), other.end());
            }

            short_vector(const short_vector& other, const Allocator& alloc)
            : _size(0), _alloc(alloc, _local_data())
            {
                assign(other.begin(), other.end());
            }

            short_vector(short_vector&& other)
            : _size(other._size), _alloc(std::move(other._alloc), _local_data())
            {
                if (other._is_local())
                {
                    _uninitialized_move_n_if(other.data(), other.size(), data());
                    other.clear();
                }
                else
                {
                    _capacity = other._capacity;
                    _alloc._data = other._alloc._data;
                    other._size = 0;
                    other._alloc._data = other._local_data();
                }
            }

            short_vector(short_vector&& other, const Allocator& alloc)
            : _size(other._size), _alloc(alloc, _local_data())
            {
                if (other._is_local() || alloc != other._alloc)
                {
                    reserve(size());
                    _uninitialized_move_n_if(other.data(), size(), data());
                    other.clear();
                }
                else
                {
                    _capacity = other._capacity;
                    _alloc._data = other._alloc._data;
                    other._size = 0;
                    other._alloc._data = other._local_data();
                }
            }

            short_vector(std::initializer_list<T> init,
                         const Allocator& alloc = Allocator())
            : _size(0), _alloc(alloc, _local_data())
            {
                assign(init.begin(), init.end());
            }

            ~short_vector()
            {
                _set_capacity(0);
            }

            short_vector& operator=(const short_vector& other)
            {
                if (_alloc_traits::propagate_on_container_copy_assignment::value)
                    _alloc = other._alloc;

                assign(other.begin(), other.end());

                return *this;
            }

            short_vector& operator=(short_vector&& other)
            {
                if (_alloc_traits::propagate_on_container_move_assignment::value)
                    _alloc = std::move(other._alloc);

                if (other._is_local() || (!_alloc_traits::propagate_on_container_move_assignment::value && _alloc != other._alloc))
                {
                    assign(other.begin(), other.end());
                    other.clear();
                }
                else if (_is_local())
                {
                    clear();
                    _size = other._size;
                    _capacity = other._capacity;
                    _alloc._data = other._alloc._data;
                    other._size = 0;
                    other._alloc._data = other._local_data();
                }
                else
                {
                    swap(_size, other._size);
                    swap(_capacity, other._capacity);
                    swap(_alloc._data, other._alloc._data);
                }

                return *this;
            }

            short_vector& operator=(std::initializer_list<T> ilist)
            {
                assign(ilist.begin(), ilist.end());
                return *this;
            }

            void assign(size_type count, const T& value)
            {
                if (capacity() < count)
                {
                    clear();
                    reserve(count);
                }

                if (size() > count)
                    erase(begin()+count, end());
                std::fill_n(begin(), std::min(count, size()), value);
                if (count > size())
                    _construct_n(end(), count-size(), value);

                _size = count;
            }

            template <typename Iterator>
            void assign(Iterator first, Iterator last)
            {
                _assign(first, last, typename std::iterator_traits<Iterator>
                                                 ::iterator_category());
            }

            void assign(std::initializer_list<T> ilist)
            {
                assign(ilist.begin(), ilist.end());
            }

            allocator_type get_allocator() const
            {
                return _alloc;
            }

            reference at(size_type pos)
            {
                if (pos >= size())
                    throw std::out_of_range("short_vector: out-of-range");
                return data()[pos];
            }

            const_reference at(size_type pos) const
            {
                if (pos >= size())
                    throw std::out_of_range("short_vector: out-of-range");
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
                return data()[size()-1];
            }

            const_reference back() const
            {
                return data()[size()-1];
            }

            pointer data()
            {
                return _alloc._data;
            }

            const_pointer data() const
            {
                return _alloc._data;
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
                return data()+size();
            }

            const_iterator end() const
            {
                return data()+size();
            }

            const_iterator cend() const
            {
                return data()+size();
            }

            reverse_iterator rbegin()
            {
                return end();
            }

            const_reverse_iterator rbegin() const
            {
                return end();
            }

            const_reverse_iterator crbegin() const
            {
                return end();
            }

            reverse_iterator rend()
            {
                return begin();
            }

            const_reverse_iterator rend() const
            {
                return begin();
            }

            const_reverse_iterator crend() const
            {
                return begin();
            }

            bool empty() const
            {
                return size() == 0;
            }

            size_type size() const
            {
                return _size;
            }

            size_type max_size() const
            {
                return std::numeric_limits<size_type>::max();
            }

            void reserve(size_type new_cap)
            {
                if (new_cap > capacity())
                    _set_capacity(_new_capacity(capacity(), new_cap));
            }

            size_type capacity() const
            {
                return (_is_local() ? N : _capacity);
            }

            void shrink_to_fit()
            {
                _set_capacity(size());
            }

            void clear()
            {
                erase(begin(), end());
            }

            iterator insert(const_iterator pos, const T& value)
            {
                return _emplace(pos, 1, value);
            }

            iterator insert(const_iterator pos, T&& value)
            {
                return _emplace(pos, 1, std::move(value));
            }

            iterator insert(const_iterator pos, size_type count, const T& value)
            {
                return _emplace(pos, count, value);
            }

            template <typename Iterator>
            iterator insert(const_iterator pos, Iterator first, Iterator last)
            {
                return _insert(pos, first, last, typename std::iterator_traits<Iterator>::iterator_category());
            }

            iterator insert(const_iterator pos, std::initializer_list<T> ilist)
            {
                return insert(pos, ilist.begin(), ilist.end());
            }

            template <typename... Args>
            iterator emplace(const_iterator pos, Args&&... args)
            {
                return _emplace(pos, 1, std::forward<Args>(args)...);
            }

            iterator erase(const_iterator pos)
            {
                return erase(pos, pos+1);
            }

            iterator erase(const_iterator first, const_iterator last)
            {
                size_type n = last-first;
                std::move(last, end(), first);
                _destroy(end()-n, end());
                _size -= n;
                return const_cast<iterator>(first);
            }

            void push_back(const T& value)
            {
                _emplace(end(), 1, value);
            }

            void push_back(T&& value)
            {
                _emplace(end(), 1, std::move(value));
            }

            template <typename... Args>
            void emplace_back(Args&&... args)
            {
                _emplace(end(), 1, std::forward<Args>(args)...);
            }

            void pop_back()
            {
                erase(end()-1);
            }

            void resize(size_type count)
            {
                resize(count, T());
            }

            void resize(size_type count, const value_type& value)
            {
                reserve(count);
                if (count < size())
                    _destroy(begin()+count, end());
                else if (count > size())
                    _construct_n(end(), count-size(), value);
                _size = count;
            }

            void swap(short_vector& other)
            {
                using std::swap;

                if (_alloc_traits::propagate_on_container_swap::value)
                    swap(_alloc, other._alloc);

                if ((_is_local() && other._is_local()) ||
                    (!_alloc_traits::propagate_on_container_swap::value && _alloc != other._alloc))
                {
                    reserve(other.size());
                    other.reserve(size());

                    std::swap_ranges(begin(), begin()+std::min(size(), other.size()), other.begin());
                    if (size() < other.size())
                    {
                        _unitialized_move_n_if(other.begin()+size(), other.size()-size(), end());
                        _destroy(other.begin()+size(), other.end());
                    }
                    else if (size() > other.size())
                    {
                        _unitialized_move_n_if(begin()+other.size(), size()-other.size(), other.end());
                        _destroy(begin()+other.size(), end());
                    }
                }
                else if (_is_local())
                {
                    size_type other_cap = other.capacity();
                    pointer other_data = other.data();
                    other._alloc._data = other._local_data();
                    _unitialized_move_n_if(begin(), size(), other.begin());
                    _destroy(begin(), end());
                    _capacity = other_cap;
                    _alloc._data = other_data;
                }
                else if (other._is_local())
                {
                    size_type this_cap = capacity();
                    pointer this_data = data();
                    _alloc._data = _local_data();
                    _unitialized_move_n_if(other.begin(), other.size(), begin());
                    _destroy(other.begin(), other.end());
                    other._capacity = this_cap;
                    other._alloc._data = this_data;
                }
                else
                {
                    swap(_capacity, other._capacity);
                    swap(_alloc._data, other._alloc._data);
                }

                swap(_size, other._size);
            }

            friend void swap(short_vector& lhs, short_vector& rhs)
            {
                lhs.swap(rhs);
            }

            friend bool operator==(const short_vector& lhs,
                                   const short_vector& rhs)
            {
                return lhs.size() == rhs.size() &&
                    std::equal(lhs.begin(), lhs.end(), rhs.begin());
            }

            friend bool operator!=(const short_vector& lhs,
                                   const short_vector& rhs)
            {
                return !(lhs == rhs);
            }

            friend bool operator<(const short_vector& lhs,
                                  const short_vector& rhs)
            {
                return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                                    rhs.begin(), rhs.end());
            }

            friend bool operator<=(const short_vector& lhs,
                                   const short_vector& rhs)
            {
                return !(rhs < lhs);
            }

            friend bool operator>(const short_vector& lhs,
                                  const short_vector& rhs)
            {
                return (rhs < lhs);
            }

            friend bool operator>=(const short_vector& lhs,
                                   const short_vector& rhs)
            {
                return !(lhs < rhs);
            }

        protected:
            void _set_capacity(size_type new_cap)
            {
                size_type old_cap = capacity();
                pointer old_data = data();

                if (new_cap <= N)
                {
                    if (_is_local()) return;
                    _alloc._data = _local_data();
                }
                else
                {
                    _capacity = new_cap;
                    _alloc._data = _alloc_traits::allocate(_alloc, _capacity);
                }

                if (new_cap < size()) _destroy(old_data+new_cap, old_data+size());
                _uninitialized_move_n_if(old_data, std::min(new_cap, size()), data());
                _alloc_traits::deallocate(_alloc, old_data, old_cap);
            }

            template <typename Iterator>
            void _destroy(iterator first, iterator last)
            {
                if (first == last) return;
                for (--last;last != first;--last)
                    _alloc_traits::destroy(_alloc, last);
            }

            template <typename Iterator>
            iterator _construct(Iterator first, Iterator last, iterator result)
            {
                iterator cur = result;

                try
                {
                    for (;first != last;++first, ++cur)
                        _alloc_traits::construct(_alloc, cur, *first);
                }
                catch (...)
                {
                    _destroy(result, cur);
                    throw;
                }

                return result;
            }

            template <typename... Args>
            iterator _construct_n(iterator result, size_type n, Args&&... args)
            {
                iterator last = result+n;
                iterator cur = result;

                try
                {
                    for (;cur != last;++cur)
                        _alloc_traits::construct(_alloc, cur, std::forward<Args>(args)...);
                }
                catch (...)
                {
                    _destroy(result, cur);
                    throw;
                }

                return result;
            }

            iterator _uninitialized_move_n_if(iterator first, size_type n, iterator result)
            {
                if (std::is_nothrow_move_constructible<T>::value ||
                    !std::is_copy_constructible<T>::value)
                {
                    auto it = std::make_move_iterator(first);
                    return _construct(it, it+n, result);
                }
                else
                {
                    return _construct(first, first+n, result);
                }
            }

            template <typename... Args>
            iterator _emplace(const_iterator pos, size_type n, Args&&... args)
            {
                size_type off = pos-begin();
                reserve(size()+n);
                pos = begin()+off;

                size_type n_tail = end()-pos;
                size_type n_move = std::min(n, n_tail);
                size_type n_ctr = (pos+n)-end();
                _uninitialized_move_n_if(end()-n_move, n_move, end()+n_ctr);
                try
                {
                    _construct_n(end(), n_ctr, std::forward<Args>(args)...);
                }
                catch (...)
                {
                    _destroy(end()+n_ctr, end()+n_ctr+n_move);
                    throw;
                }

                n_move = n_tail-n_move;
                n_ctr = n-n_ctr;
                try
                {
                    std::move_backward(pos, pos+n_move, end());
                    std::fill_n(pos, n_ctr, T(std::forward<Args>(args)...));
                }
                catch (...)
                {
                    _destroy(end(), end()+n);
                    throw;
                }

                _size += n;
                return begin()+off+n;
            }

            template <typename Iterator>
            iterator _insert(const_iterator pos, Iterator first, Iterator last,
                             std::input_iterator_tag)
            {
                for (;first != last;++first)
                {
                    pos = _emplace(pos, 1, *first);
                }
            }

            template <typename Iterator>
            void _insert(const_iterator pos, Iterator first, Iterator last,
                         std::random_access_iterator_tag)
            {
                size_type n = last-first;
                size_type off = pos-begin();
                reserve(size()+n);
                pos = begin()+off;

                size_type n_tail = end()-pos;
                size_type n_move = std::min(n, n_tail);
                size_type n_ctr = (pos+n)-end();
                _uninitialized_move_n_if(end()-n_move, n_move, end()+n_ctr);
                try
                {
                    _construct(last-n_ctr, last, end());
                }
                catch (...)
                {
                    _destroy(end()+n_ctr, end()+n_ctr+n_move);
                    throw;
                }

                n_move = n_tail-n_move;
                n_ctr = n-n_ctr;
                try
                {
                    std::move_backward(pos, pos+n_move, end());
                    std::fill(first, first+n_ctr, pos);
                }
                catch (...)
                {
                    _destroy(end(), end()+n);
                    throw;
                }

                _size += n;
                return begin()+off+n;
            }

            static size_type _new_capacity(size_type capacity, size_type count)
            {
                return std::max(2*capacity, count);
            }

            pointer _local_data()
            {
                return &_fixed[0];
            }

            const_pointer _local_data() const
            {
                return &_fixed[0];
            }

            bool _is_local() const
            {
                return _alloc._data == _local_data();
            }

            size_type _size;
            struct _data_s : public Allocator
            {
                _data_s(Allocator alloc, pointer data)
                : Allocator(alloc), _data(data) {}

                _data_s(const _data_s& other, pointer data)
                : Allocator(other), _data(data) {}

                _data_s(_data_s&& other, pointer data)
                : Allocator(std::move(other)), _data(data) {}

                _data_s& operator=(const _data_s& other)
                {
                    Allocator::operator=(other);
                    return *this;
                }

                _data_s& operator=(_data_s&& other)
                {
                    Allocator::operator=(std::move(other));
                    return *this;
                }

                pointer _data;
            } _alloc;

            union
            {
                std::array<T,N> _fixed;
                size_type _capacity;
            };
    };
}

#endif
