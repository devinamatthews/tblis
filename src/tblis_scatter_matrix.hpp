#ifndef _TBLIS_SCATTER_MATRIX_HPP_
#define _TBLIS_SCATTER_MATRIX_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
class const_scatter_matrix_view;

template <typename T>
class scatter_matrix_view;

template <typename T, unsigned ndim> void copy(const_scatter_matrix_view<T> a, scatter_matrix_view<T> b);

namespace detail
{
    template <typename T>
    class const_scatter_matrix_ref;

    template <typename T>
    class scatter_matrix_ref;

    template <typename T>
    class const_scatter_matrix_slice;

    template <typename T>
    class scatter_matrix_slice;

    template <typename T>
    class const_scatter_matrix_ref
    {
        template <typename T_> friend class tblis::const_scatter_matrix_view;
        template <typename T_> friend class tblis::scatter_matrix_view;
        template <typename T_> friend class const_scatter_matrix_ref;
        template <typename T_> friend class scatter_matrix_ref;
        template <typename T_> friend class const_scatter_matrix_slice;
        template <typename T_> friend class scatter_matrix_slice;

        protected:
            typedef scatter_matrix_view<T> base;

            typedef typename base::stride_type stride_type;
            typedef typename base::scatter_type scatter_type;
            typedef typename base::idx_type idx_type;
            typedef typename base::value_type value_type;
            typedef typename base::pointer pointer;
            typedef typename base::const_pointer const_pointer;
            typedef typename base::reference reference;
            typedef typename base::const_reference const_reference;

            base& array;
            stride_type idx;

            const_scatter_matrix_ref(const const_scatter_matrix_ref& other) = default;

            const_scatter_matrix_ref(const base& array, idx_type i)
            : array(const_cast<base&>(static_cast<const base&>(array))),
              idx(array.stride_[1] == 0 ? array.scatter_[1][i] : i*array.stride_[1]) {}

            const_scatter_matrix_ref& operator=(const const_scatter_matrix_ref&) = delete;

        public:
            const_reference operator[](idx_type i) const
            {
                return data()[array.stride_[0] == 0 ? array.scatter_[0][i] : i*array.stride_[0]];
            }

            const_reference operator()(idx_type i) const
            {
                return (*this)[i];
            }

            const_pointer data() const
            {
                return array.data_+idx;
            }
    };

    template <typename T>
    class scatter_matrix_ref : public const_scatter_matrix_ref<T>
    {
        template <typename T_> friend class tblis::const_scatter_matrix_view;
        template <typename T_> friend class tblis::scatter_matrix_view;
        template <typename T_> friend class const_scatter_matrix_ref;
        template <typename T_> friend class scatter_matrix_ref;
        template <typename T_> friend class const_scatter_matrix_slice;
        template <typename T_> friend class scatter_matrix_slice;

        protected:
            typedef scatter_matrix_view<T> base;
            typedef const_scatter_matrix_ref<T> parent;

            typedef typename base::stride_type stride_type;
            typedef typename base::scatter_type scatter_type;
            typedef typename base::idx_type idx_type;
            typedef typename base::value_type value_type;
            typedef typename base::pointer pointer;
            typedef typename base::const_pointer const_pointer;
            typedef typename base::reference reference;
            typedef typename base::const_reference const_reference;

            using parent::array;
            using parent::idx;

            scatter_matrix_ref(const parent& other)
            : parent(other) {}

            scatter_matrix_ref(const scatter_matrix_ref& other) = default;

            scatter_matrix_ref(scatter_matrix_view<T>& array, stride_type idx, idx_type i)
            : parent(array, idx, i) {}

            scatter_matrix_ref& operator=(const scatter_matrix_ref&) = delete;

        public:
            using parent::operator[];

            reference operator[](idx_type i)
            {
                return const_cast<reference>(parent::operator[](i));
            }

            using parent::operator();

            reference operator()(idx_type i)
            {
                return (*this)[i];
            }

            using parent::data;

            pointer data()
            {
                return const_cast<pointer>(parent::data());
            }
    };

    template <typename T>
    class const_scatter_matrix_slice
    {
        template <typename T_> friend class tblis::const_scatter_matrix_view;
        template <typename T_> friend class tblis::scatter_matrix_view;
        template <typename T_> friend class const_scatter_matrix_ref;
        template <typename T_> friend class scatter_matrix_ref;
        template <typename T_> friend class const_scatter_matrix_slice;
        template <typename T_> friend class scatter_matrix_slice;

        protected:
            typedef scatter_matrix_view<T> base;

            typedef typename base::stride_type stride_type;
            typedef typename base::scatter_type scatter_type;
            typedef typename base::idx_type idx_type;
            typedef typename base::value_type value_type;
            typedef typename base::pointer pointer;
            typedef typename base::const_pointer const_pointer;
            typedef typename base::reference reference;
            typedef typename base::const_reference const_reference;

            base& array;
            stride_type idx;
            idx_type len;
            scatter_type cscat;

            const_scatter_matrix_slice(const const_scatter_matrix_slice& other) = default;

            template <typename I>
            const_scatter_matrix_slice(const const_scatter_matrix_view<T>& array, const range_t<I>& r)
            : array(const_cast<base&>(static_cast<const base&>(array))),
              idx(array.stride_[1] == 0 ? array.scatter_[1][r.front()] : r.front()*array.stride_[1]),
              len(r.size()), cscat(array.stride_[1] == 0 ? array.scatter_[1]+r.front() : nullptr) {}

            const_scatter_matrix_slice& operator=(const const_scatter_matrix_slice&) = delete;

        public:

            template <typename I>
            const_scatter_matrix_view<T> operator[](const range_t<I>& x) const
            {
                assert(x.front() <= x.back() && x.front() >= 0 && x.back() <= array.len_[0]);

                if (array.stride_[0] == 0)
                {
                    if (array.stride_[1] == 0)
                        return {x.size(), len, data()+array.scatter_[0][x.front()], array.scatter_[0]+x.front(), cscat};
                    else
                        return {x.size(), len, data()+array.scatter_[0][x.front()], array.scatter_[0]+x.front(), array.stride_[1]};
                }
                else
                {
                    if (array.stride_[1] == 0)
                        return {x.size(), len, data()+x.front()*array.stride_[0], array.stride_[0], cscat};
                    else
                        return {x.size(), len, data()+x.front()*array.stride_[0], array.stride_[0], array.stride_[1]};
                }
            }

            const_scatter_matrix_view<T> operator[](MArray::slice::all_t x) const
            {
                return *this;
            }

            template <typename I>
            const_scatter_matrix_view<T> operator()(const range_t<I>& x) const
            {
                return (*this)[x];
            }

            const_scatter_matrix_view<T> operator()(MArray::slice::all_t x) const
            {
                return (*this)[x];
            }

            const_pointer data() const
            {
                return array.data_+idx;
            }

            operator const_scatter_matrix_view<T>() const
            {
                return (*this)[range(array.len_[0])];
            }
    };

    template <typename T>
    class scatter_matrix_slice : public const_scatter_matrix_slice<T>
    {
        template <typename T_> friend class tblis::const_scatter_matrix_view;
        template <typename T_> friend class tblis::scatter_matrix_view;
        template <typename T_> friend class const_scatter_matrix_ref;
        template <typename T_> friend class scatter_matrix_ref;
        template <typename T_> friend class const_scatter_matrix_slice;
        template <typename T_> friend class scatter_matrix_slice;

        protected:
            typedef scatter_matrix_view<T> base;
            typedef const_scatter_matrix_slice<T> parent;

            typedef typename base::stride_type stride_type;
            typedef typename base::scatter_type scatter_type;
            typedef typename base::idx_type idx_type;
            typedef typename base::value_type value_type;
            typedef typename base::pointer pointer;
            typedef typename base::const_pointer const_pointer;
            typedef typename base::reference reference;
            typedef typename base::const_reference const_reference;

            using parent::array;
            using parent::idx;
            using parent::len;
            using parent::cscat;

            scatter_matrix_slice(const parent& other)
            : parent(other) {}

            scatter_matrix_slice(const scatter_matrix_slice& other) = default;

            template <typename I>
            scatter_matrix_slice(scatter_matrix_view<T>& array, const range_t<I>& r)
            : parent(array, r) {}

        public:
            scatter_matrix_slice& operator=(const parent& other)
            {
                copy(view(other), view(*this));
                return *this;
            }

            scatter_matrix_slice& operator=(const scatter_matrix_slice& other)
            {
                copy(view(other), view(*this));
                return *this;
            }

            using parent::operator[];

            template <typename I>
            scatter_matrix_view<T> operator[](const range_t<I>& x)
            {
                return {parent::operator[](x)};
            }

            scatter_matrix_view<T> operator[](MArray::slice::all_t x)
            {
                return {parent::operator[](x)};
            }

            using parent::operator();

            template <typename I>
            scatter_matrix_view<T> operator()(const range_t<I>& x)
            {
                return (*this)[x];
            }

            scatter_matrix_view<T> operator()(MArray::slice::all_t x)
            {
                return (*this)[x];
            }

            using parent::data;

            pointer data()
            {
                return const_cast<pointer>(parent::data());
            }

            using parent::operator const_scatter_matrix_view<T>;

            operator scatter_matrix_view<T>()
            {
                return {parent::operator const_scatter_matrix_view<T>()};
            }
    };

    template <typename T>
    const_scatter_matrix_view<T> view(const const_scatter_matrix_slice<T>& x)
    {
        return x;
    }

    template <typename T>
    scatter_matrix_view<T> view(scatter_matrix_slice<T>& x)
    {
        return x;
    }

    template <typename T>
    scatter_matrix_view<T> view(scatter_matrix_slice<T>&& x)
    {
        return x;
    }
}

template <typename T>
class const_scatter_matrix_view
{
    template <typename T_> friend class const_scatter_matrix_view;
    template <typename T_> friend class scatter_matrix_view;
    template <typename T_> friend class const_scatter_matrix_ref;
    template <typename T_> friend class scatter_matrix_ref;
    template <typename T_> friend class const_scatter_matrix_slice;
    template <typename T_> friend class scatter_matrix_slice;

    public:
        typedef unsigned idx_type;
        typedef size_t size_type;
        typedef ptrdiff_t stride_type;
        typedef const stride_type* scatter_type;
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;

    protected:
        pointer data_ = nullptr;
        std::array<idx_type,2> len_ = {};
        std::array<stride_type,2> stride_ = {};
        std::array<scatter_type,2> scatter_ = {};

        const_scatter_matrix_view& operator=(const const_scatter_matrix_view& other) = delete;

    public:
        const_scatter_matrix_view() {}

        const_scatter_matrix_view(const const_scatter_matrix_view<T>& other)
        {
            reset(other);
        }

        const_scatter_matrix_view(const scatter_matrix_view<T>& other)
        {
            reset(other);
        }

        const_scatter_matrix_view(idx_type m, idx_type n, const_pointer ptr, stride_type rs, stride_type cs)
        {
            reset(m, n, ptr, rs, cs);
        }

        const_scatter_matrix_view(idx_type m, idx_type n, const_pointer ptr, scatter_type rscat, stride_type cs)
        {
            reset(m, n, ptr, rscat, cs);
        }

        const_scatter_matrix_view(idx_type m, idx_type n, const_pointer ptr, stride_type rs, scatter_type cscat)
        {
            reset(m, n, ptr, rs, cscat);
        }

        const_scatter_matrix_view(idx_type m, idx_type n, const_pointer ptr, scatter_type rscat, scatter_type cscat)
        {
            reset(m, n, ptr, rscat, cscat);
        }

        void reset()
        {
            data_ = nullptr;
            len_.fill(0);
            stride_.fill(0);
        }

        void reset(const const_scatter_matrix_view<T>& other)
        {
            data_ = other.data_;
            len_ = other.len_;
            stride_ = other.stride_;
        }

        void reset(const scatter_matrix_view<T>& other)
        {
            reset(static_cast<const const_scatter_matrix_view<T>&>(other));
        }

        void reset(idx_type m, idx_type n, const_pointer ptr, stride_type rs, stride_type cs)
        {
            data_ = const_cast<pointer>(ptr);
            len_[0] = m;
            len_[1] = n;
            stride_[0] = rs;
            stride_[1] = cs;
            scatter_[0] = nullptr;
            scatter_[1] = nullptr;
        }

        void reset(idx_type m, idx_type n, const_pointer ptr, scatter_type rscat, stride_type cs)
        {
            data_ = const_cast<pointer>(ptr);
            len_[0] = m;
            len_[1] = n;
            stride_[0] = 0;
            stride_[1] = cs;
            scatter_[0] = rscat;
            scatter_[1] = nullptr;
        }

        void reset(idx_type m, idx_type n, const_pointer ptr, stride_type rs, scatter_type cscat)
        {
            data_ = const_cast<pointer>(ptr);
            len_[0] = m;
            len_[1] = n;
            stride_[0] = rs;
            stride_[1] = 0;
            scatter_[0] = nullptr;
            scatter_[1] = cscat;
        }

        void reset(idx_type m, idx_type n, const_pointer ptr, scatter_type rscat, scatter_type cscat)
        {
            data_ = const_cast<pointer>(ptr);
            len_[0] = m;
            len_[1] = n;
            stride_[0] = 0;
            stride_[1] = 0;
            scatter_[0] = rscat;
            scatter_[1] = cscat;
        }

        void shift_down(unsigned dim, idx_type n)
        {
            assert(dim < 2);

            if (stride_[dim] == 0)
            {
                scatter_[dim] += n;
            }
            else
            {
                data_ += n*stride_[dim];
            }
        }

        void shift_up(unsigned dim, idx_type n)
        {
            assert(dim < 2);

            if (stride_[dim] == 0)
            {
                scatter_[dim] -= n;
            }
            else
            {
                data_ -= n*stride_[dim];
            }
        }

        void shift_down(unsigned dim)
        {
            shift_down(dim, len_[dim]);
        }

        void shift_up(unsigned dim)
        {
            shift_up(dim, len_[dim]);
        }

        const_scatter_matrix_view<T> shifted_down(unsigned dim, idx_type n) const
        {
            const_scatter_matrix_view<T> r(*this);
            r.shift_down(dim, n);
            return r;
        }

        const_scatter_matrix_view<T> shifted_up(unsigned dim, idx_type n) const
        {
            const_scatter_matrix_view<T> r(*this);
            r.shift_up(dim, n);
            return r;
        }

        const_scatter_matrix_view<T> shifted_down(unsigned dim) const
        {
            return shifted_down(dim, len_[dim]);
        }

        const_scatter_matrix_view<T> shifted_up(unsigned dim) const
        {
            return shifted_up(dim, len_[dim]);
        }

        template <typename U>
        void permute(const std::array<U, 2>& perm)
        {
            assert((perm[0] == 0 && perm[1] == 1) ||
                   (perm[0] == 1 && perm[1] == 0));

            if (perm[0] == 1)
            {
                using std::swap;
                swap(len_[0], len_[1]);
                swap(stride_[0], stride_[1]);
                swap(scatter_[0], scatter_[1]);
            }
        }

        template <typename U>
        const_scatter_matrix_view<T> permuted(const std::array<U, 2>& perm) const
        {
            const_scatter_matrix_view<T> r(*this);
            r.permute(perm);
            return r;
        }

        void permute(unsigned p0, unsigned p1)
        {
            permute(make_array(p0, p1));
        }

        const_scatter_matrix_view<T> permuted(unsigned p0, unsigned p1) const
        {
            return permuted(make_array(p0, p1));
        }

        void transpose()
        {
            permute(1, 0);
        }

        const_scatter_matrix_view<T> transposed() const
        {
            return permuted(1, 0);
        }

        detail::const_scatter_matrix_ref<T> operator[](idx_type i) const
        {
            assert(i < len_[0]);
            return {*this, i};
        }

        template <typename I>
        detail::const_scatter_matrix_slice<T> operator[](const range_t<I>& x) const
        {
            assert(x.front() >= 0 && x.back() <= len_[0]);
            return {*this, x};
        }

        detail::const_scatter_matrix_slice<T> operator[](MArray::slice::all_t x) const
        {
            return {*this, range(len_[0])};
        }

        template <typename I0, typename I1, typename=
            stl_ext::enable_if_t<MArray::detail::are_indices_or_slices<I0, I1>::value>>
        auto operator()(I0&& i0, I1&& i1) const ->
        decltype((*this)[i0][i1])
        {
            return (*this)[i0][i1];
        }

        const_pointer data() const
        {
            return data_;
        }

        idx_type length(unsigned dim) const
        {
            assert(dim < 2);
            return len_[dim];
        }

        idx_type length(unsigned dim, idx_type len)
        {
            assert(dim < 2);
            std::swap(len, len_[dim]);
            return len;
        }

        const std::array<idx_type, 2>& lengths() const
        {
            return len_;
        }

        stride_type stride(unsigned dim) const
        {
            assert(dim < 2);
            return stride_[dim];
        }

        stride_type stride(unsigned dim, stride_type stride)
        {
            assert(dim < 2);
            std::swap(stride, stride_[dim]);
            scatter_[dim] = nullptr;
            return stride;
        }

        const std::array<stride_type, 2>& strides() const
        {
            return stride_;
        }

        scatter_type scatter(unsigned dim) const
        {
            assert(dim < 2);
            return scatter_[dim];
        }

        void scatter(unsigned dim, scatter_type scatter)
        {
            assert(dim < 2);
            std::swap(scatter, scatter_[dim]);
            stride_[dim] = 0;
        }

        const std::array<scatter_type, 2>& scatters() const
        {
            return scatter_;
        }

        unsigned dimension() const
        {
            return 2;
        }
};

template <typename T>
class scatter_matrix_view : protected const_scatter_matrix_view<T>
{
    protected:
        typedef const_scatter_matrix_view<T> base;

    public:
        using typename base::idx_type;
        using typename base::size_type;
        using typename base::stride_type;
        using typename base::scatter_type;
        using typename base::value_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;

    protected:
        using base::data_;
        using base::len_;
        using base::stride_;

        scatter_matrix_view(const base& other)
        : base(other) {}

    public:
        scatter_matrix_view() {}

        scatter_matrix_view(scatter_matrix_view<T>& other)
        : base(other) {}

        scatter_matrix_view(scatter_matrix_view<T>&& other)
        : base(other) {}

        scatter_matrix_view(idx_type m, idx_type n, pointer ptr, stride_type rs, stride_type cs)
        {
            reset(m, n, ptr, rs, cs);
        }

        scatter_matrix_view(idx_type m, idx_type n, pointer ptr, scatter_type rscat, stride_type cs)
        {
            reset(m, n, ptr, rscat, cs);
        }

        scatter_matrix_view(idx_type m, idx_type n, pointer ptr, stride_type rs, scatter_type cscat)
        {
            reset(m, n, ptr, rs, cscat);
        }

        scatter_matrix_view(idx_type m, idx_type n, pointer ptr, scatter_type rscat, scatter_type cscat)
        {
            reset(m, n, ptr, rscat, cscat);
        }

        void reset()
        {
            base::reset();
        }

        void reset(scatter_matrix_view<T>& other)
        {
            base::reset(other);
        }

        void reset(scatter_matrix_view<T>&& other)
        {
            base::reset(other);
        }

        void reset(idx_type m, idx_type n, pointer ptr, stride_type rs, stride_type cs)
        {
            base::reset(m, n, ptr, rs, cs);
        }

        void reset(idx_type m, idx_type n, pointer ptr, scatter_type rscat, stride_type cs)
        {
            base::reset(m, n, ptr, rscat, cs);
        }

        void reset(idx_type m, idx_type n, pointer ptr, stride_type rs, scatter_type cscat)
        {
            base::reset(m, n, ptr, rs, cscat);
        }

        void reset(idx_type m, idx_type n, pointer ptr, scatter_type rscat, scatter_type cscat)
        {
            base::reset(m, n, ptr, rscat, cscat);
        }

        scatter_matrix_view& operator=(const const_scatter_matrix_view<T>& other) const
        {
            copy(other, *this);
            return *this;
        }

        scatter_matrix_view& operator=(const scatter_matrix_view<T>& other) const
        {
            return operator=(static_cast<const const_scatter_matrix_view<T>&>(other));
        }

        scatter_matrix_view& operator=(const T& value) const
        {
            for (idx_type j = 0;j < len_[1];j++)
            {
                for (idx_type i = 0;i < len_[0];i++)
                {
                    (*this)[i][j] = value;
                }
            }

            return *this;
        }

        using base::shift_down;
        using base::shift_up;

        scatter_matrix_view<T> shifted_down(unsigned dim, idx_type n) const
        {
            return base::shifted_down(dim, n);
        }

        scatter_matrix_view<T> shifted_up(unsigned dim, idx_type n) const
        {
            return base::shifted_up(dim, n);
        }

        scatter_matrix_view<T> shifted_down(unsigned dim) const
        {
            return base::shifted_down(dim);
        }

        scatter_matrix_view<T> shifted_up(unsigned dim) const
        {
            return base::shifted_up(dim);
        }

        using base::permute;

        template <typename U>
        scatter_matrix_view<T> permuted(const std::array<U, 2>& perm) const
        {
            return base::permuted(perm);
        }

        scatter_matrix_view<T> permuted(unsigned p0, unsigned p1) const
        {
            return base::permuted(p0, p1);
        }

        using base::transpose;

        scatter_matrix_view<T> transposed() const
        {
            return base::transposed();
        }

        void rotate_dim(unsigned dim, stride_type shift) const
        {
            assert(dim < 2);
            abort();
            //TODO
        }

        template <typename U>
        void rotate(const std::array<U, 2>& shift) const
        {
            rotate_dim(0, shift[0]);
            rotate_dim(1, shift[1]);
        }

        void rotate(stride_type s0, stride_type s1) const
        {
            rotate(make_array(s0, s1));
        }

        detail::scatter_matrix_ref<T> operator[](idx_type i) const
        {
            return base::operator[](i);
        }

        template <typename I>
        detail::scatter_matrix_slice<T> operator[](const range_t<I>& x) const
        {
            return base::operator[](x);
        }

        detail::scatter_matrix_slice<T> operator[](MArray::slice::all_t x) const
        {
            return base::operator[](x);
        }

        template <typename I0, typename I1, typename=
            stl_ext::enable_if_t<MArray::detail::are_indices_or_slices<I0, I1>::value>>
        auto operator()(I0&& i0, I1&& i1) const ->
        decltype((*this)[i0][i1])
        {
            return (*this)[i0][i1];
        }

        pointer data() const
        {
            return const_cast<pointer>(base::data());
        }

        using base::length;
        using base::lengths;
        using base::stride;
        using base::strides;
        using base::scatter;
        using base::scatters;
        using base::dimension;
};

template <typename T, unsigned ndim> void copy(const_scatter_matrix_view<T> a, scatter_matrix_view<T> b)
{
    typedef typename const_scatter_matrix_view<T>::idx_type idx_type;

    assert(a.lengths() == b.lengths());

    for (idx_type j = 0;j < a.length(1);j++)
    {
        for (idx_type i = 0;i < a.length(0);i++)
        {
            b[i][j] = a[i][j];
        }
    }
}

}

#endif
