#ifndef _MARRAY_DPD_MARRAY_BASE_HPP_
#define _MARRAY_DPD_MARRAY_BASE_HPP_

#include "marray_view.hpp"

namespace MArray
{

template <typename Type, unsigned NDim, typename Derived, bool Owner>
class dpd_marray_base;

template <typename Type, unsigned NDim>
class dpd_marray_view;

template <typename Type, unsigned NDim, typename Allocator=std::allocator<Type>>
class dpd_marray;

template <typename Type, typename Derived, bool Owner>
class dpd_varray_base;

template <typename Type>
class dpd_varray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class dpd_varray;

namespace detail
{

template <typename U, typename V>
void set_size(unsigned irrep, const U& len, V& size, dpd_layout layout)
{
    if (layout == BALANCED_ROW_MAJOR ||
        layout == BALANCED_COLUMN_MAJOR)
    {
        set_size_balanced(irrep, len, size, 0, len.length(0), 0);
    }
    else
    {
        set_size_blocked(irrep, len, size);
    }
}

template <typename U, typename V>
void set_size_balanced(unsigned irrep, const U& len, V& size,
                       unsigned begin, unsigned end, unsigned idx)
{
    unsigned nirrep = len.length(1);
    unsigned ndim = end-begin;

    if (ndim == 1)
    {
        size[idx][irrep] = len[begin][irrep];
        return;
    }

    unsigned mid = begin+(ndim+1)/2;
    unsigned idxl = idx+1;
    unsigned idxr = idx+((ndim+1)/2)*2;

    stride_type local_size = 0;
    for (unsigned irrepr = 0;irrepr < nirrep;irrepr++)
    {
        unsigned irrepl = irrepr^irrep;
        set_size_balanced(irrepl, len, size, begin, mid, idxl);
        set_size_balanced(irrepr, len, size,   mid, end, idxr);
        local_size += size[idxl][irrepl]*size[idxr][irrepr];
    }

    size[idx][irrep] = local_size;
}

template <typename U, typename V>
void set_size_blocked(unsigned irrep, const U& len, V& size)
{
    unsigned ndim = len.length(0);
    unsigned nirrep = len.length(1);

    size[0] = len[0];

    for (unsigned i = 1;i < ndim;i++)
    {
        for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
        {
            size[i][irr1] = 0;
            for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
            {
                size[i][irr1] += size[i-1][irr1^irr2]*len[i][irr2];
            }
        }
    }
}

template <typename U, typename V, typename W, typename X, typename Y, typename Z, typename Q>
void get_block(const U& iperm, const V& irreps, const W& len, const X& size,
               dpd_layout layout, Y& blen, Z& bdata, Q& bstride,
               stride_type size_before=1)
{
    if (layout == BALANCED_ROW_MAJOR ||
        layout == BALANCED_COLUMN_MAJOR)
    {
        get_block_balanced(iperm, irreps, len, size, blen, bdata, bstride,
                           size_before, 0, len.length(0), 0);
    }
    else if (layout == PREFIX_ROW_MAJOR ||
             layout == PREFIX_COLUMN_MAJOR)
    {
        get_block_prefix(iperm, irreps, len, size, blen, bdata, bstride, size_before);
    }
    else
    {
        get_block_blocked(iperm, irreps, len, size, blen, bdata, bstride, size_before);
    }
}

template <typename U, typename V, typename W, typename X, typename Y, typename Z, typename Q>
unsigned get_block_balanced(const U& iperm, const V& irreps, const W& len, const X& size,
                            Y& blen, Z& bdata, Q& bstride, stride_type size_before,
                            unsigned begin, unsigned end, unsigned idx)
{
    unsigned ndim = end-begin;

    if (ndim == 1)
    {
        blen[iperm[begin]] = len[begin][irreps[iperm[begin]]];
        bstride[iperm[begin]] = size_before;
        return irreps[iperm[begin]];
    }

    unsigned mid = begin+(ndim+1)/2;
    unsigned idxl = idx+1;
    unsigned idxr = idx+((ndim+1)/2)*2;

    unsigned irrepl = get_block_balanced(iperm, irreps, len, size,
                                         blen, bdata, bstride, size_before,
                                         begin, mid, idxl);
    unsigned irrepr = get_block_balanced(iperm, irreps, len, size,
                                         blen, bdata, bstride, size_before*size[idxl][irrepl],
                                         mid, end, idxr);
    unsigned irrep = irrepl^irrepr;

    for (unsigned irr2 = 0;irr2 < irrepr;irr2++)
    {
        unsigned irr1 = irr2^irrep;
        bdata += size_before*size[idxl][irr1]*size[idxr][irr2];
    }

    return irrep;
}

template <typename U, typename V, typename W, typename X, typename Y, typename Z, typename Q>
void get_block_prefix(const U& iperm, const V& irreps, const W& len, const X& size,
                      Y& blen, Z& bdata, Q& bstride, stride_type size_before)
{
    unsigned ndim = len.length(0);

    unsigned lirrep = 0;
    bstride[iperm[0]] = 1;
    blen[iperm[0]] = len[0][irreps[iperm[0]]];
    for (unsigned i = 1;i < ndim;i++)
    {
        lirrep ^= irreps[iperm[i-1]];
        bstride[iperm[i]] = size[i-1][lirrep];
        blen[iperm[i]] = len[i][irreps[iperm[i]]];
    }

    unsigned rirrep = lirrep^irreps[iperm[ndim-1]];
    for (unsigned i = ndim;i --> 1;)
    {
        for (unsigned irr1 = 0;irr1 < irreps[iperm[i]];irr1++)
        {
            unsigned irr2 = irr1^rirrep;
            bdata += size[i-1][irr2]*len[i][irr1];
        }
        rirrep ^= irreps[iperm[i]];
    }
}

template <typename U, typename V, typename W, typename X, typename Y, typename Z, typename Q>
void get_block_blocked(const U& iperm, const V& irreps, const W& len, const X& size,
                       Y& blen, Z& bdata, Q& bstride, stride_type size_before)
{
    unsigned ndim = len.length(0);

    unsigned rirrep = 0;
    stride_type lsize = 1;
    for (unsigned i = 0;i < ndim;i++)
    {
        rirrep ^= irreps[iperm[i]];
        bstride[iperm[i]] = lsize;
        lsize *= blen[iperm[i]] = len[i][irreps[iperm[i]]];
    }

    stride_type rsize = 1;
    for (unsigned i = ndim;i --> 1;)
    {
        stride_type offset = 0;
        for (unsigned irr1 = 0;irr1 < irreps[iperm[i]];irr1++)
        {
            unsigned irr2 = irr1^rirrep;
            offset += size[i-1][irr2]*len[i][irr1];
        }
        bdata += offset*rsize;
        rsize *= len[iperm[i]];
        rirrep ^= irreps[iperm[i]];
    }
}

static unsigned num_sizes(unsigned ndim, dpd_layout layout)
{
    if (layout == BALANCED_ROW_MAJOR ||
        layout == BALANCED_COLUMN_MAJOR) return 2*ndim-1;
    else return ndim;
}

}

template <typename Type, unsigned NDim, typename Derived, bool Owner>
class dpd_marray_base
{
    static_assert(NDim > 0, "NDim must be positive");

    template <typename, unsigned, typename, bool> friend class dpd_marray_base;
    template <typename, unsigned> friend class dpd_marray_view;
    template <typename, unsigned, typename> friend class dpd_marray;

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
        std::array<std::array<len_type,8>, NDim> len_ = {};
        std::array<std::array<stride_type,8>, 2*NDim> size_ = {};
        std::array<unsigned, NDim> perm_ = {};
        pointer data_ = nullptr;
        unsigned irrep_ = 0;
        unsigned nirrep_ = 0;
        dpd_layout layout_ = DEFAULT;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            len_ = {};
            size_ = {};
            perm_ = {};
            data_ = nullptr;
            irrep_ = 0;
            nirrep_ = 0;
            layout_ = DEFAULT;
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename dpd_marray_base<U, NDim, D, O>::cptr,pointer>>
        void reset(const dpd_marray_base<U, NDim, D, O>& other)
        {
            reset(const_cast<dpd_marray_base<U, NDim, D, O>&>(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename dpd_marray_base<U, NDim, D, O>::pointer,pointer>>
        void reset(dpd_marray_base<U, NDim, D, O>& other)
        {
            len_ = other.len_;
            size_ = other.size_;
            perm_ = other.perm_;
            data_ = other.data_;
            irrep_ = other.irrep_;
            nirrep_ = other.nirrep_;
            layout_ = other.layout_;
        }

        template <typename U, typename=
            detail::enable_if_assignable_t<len_type&,U>>
        void reset(unsigned irrep, unsigned nirrep,
                   initializer_matrix<U> len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            reset<initializer_matrix<U>>(irrep, nirrep, len, ptr, layout);
        }

        template <typename U, typename=
            detail::enable_if_container_of_t<U,len_type>>
        void reset(unsigned irrep, unsigned nirrep,
                   std::initializer_list<U> len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            reset<std::initializer_list<U>>(irrep, nirrep, len, ptr, layout);
        }

        template <typename U, typename=
            detail::enable_if_container_of_containers_of_t<U,len_type>>
        void reset(unsigned irrep, unsigned nirrep, const U& len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            data_ = ptr;
            irrep_ = irrep;
            nirrep_ = nirrep;
            layout_ = layout;

            if (layout == BLOCKED_COLUMN_MAJOR ||
                layout == PREFIX_COLUMN_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                // Column major
                auto it = len.begin();
                for (unsigned i = 0;i < NDim;i++)
                {
                    std::copy_n(it->begin(), nirrep, len_[i].begin());
                    perm_[i] = i;
                    ++it;
                }
            }
            else
            {
                // Row major: reverse the dimensions and treat as
                // permuted column major
                auto it = len.begin();
                for (unsigned i = 0;i < NDim;i++)
                {
                    std::copy_n(it->begin(), nirrep, len_[NDim-1-i].begin());
                    perm_[i] = NDim-1-i;
                    ++it;
                }
            }

            detail::set_size(irrep_, len_, size_, layout_);
        }

        template <typename U, typename=
            detail::enable_if_assignable_t<len_type&,U>>
        void reset(unsigned irrep, unsigned nirrep,
                   matrix_view<U> len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            data_ = ptr;
            irrep_ = irrep;
            nirrep_ = nirrep;
            layout_ = layout;

            if (layout == BLOCKED_COLUMN_MAJOR ||
                layout == PREFIX_COLUMN_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                // Column major
                for (unsigned i = 0;i < NDim;i++)
                {
                    for (unsigned j = 0;j < NDim;j++)
                        len_[i][j] = len[i][j];
                    perm_[i] = i;
                }
            }
            else
            {
                // Row major: reverse the dimensions and treat as
                // permuted column major
                for (unsigned i = 0;i < NDim;i++)
                {
                    for (unsigned j = 0;j < NDim;j++)
                        len_[NDim-1-i][j] = len[i][j];
                    perm_[i] = NDim-1-i;
                }
            }

            detail::set_size(irrep_, len_, size_, layout_);
        }

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename View, typename Func, unsigned... I>
        void for_each_block(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            typedef typename View::pointer Ptr;

            std::array<unsigned, NDim-1> nirrep;
            nirrep.fill(nirrep_);

            const_pointer cptr;
            std::array<unsigned, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            miterator<NDim-1, 0> it(nirrep);
            while (it.next())
            {
                irreps[0] = irrep_;
                for (unsigned i = 1;i < NDim;i++)
                {
                    irreps[0] ^= irreps[i] = it.position()[i-1];
                }

                get_block(irreps, len, cptr, stride);

                f(View(len, const_cast<Ptr>(cptr), stride), irreps[I]...);
            }
        }

        template <typename Tp, typename Func, unsigned... I>
        void for_each_element(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            typedef Tp* Ptr;

            std::array<unsigned, NDim-1> nirrep;
            nirrep.fill(nirrep_);

            const_pointer cptr;
            std::array<unsigned, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            miterator<NDim-1, 0> it(nirrep);
            while (it.next())
            {
                irreps[0] = irrep_;
                for (unsigned i = 1;i < NDim;i++)
                {
                    irreps[0] ^= irreps[i] = it.position()[i-1];
                }

                get_block(irreps, len, cptr, stride);

                miterator<NDim, 1> it(len, stride);
                Ptr ptr = const_cast<Ptr>(cptr);
                while (it.next(ptr)) f(*ptr, irreps[I]..., it.position()[I]...);
            }
        }

        void swap(dpd_marray_base& other)
        {
            using std::swap;
            swap(len_, other.len_);
            swap(size_, other.size_);
            swap(perm_, other.perm_);
            swap(data_, other.data_);
            swap(irrep_, other.irrep_);
            swap(nirrep_, other.nirrep_);
            swap(layout_, other.layout_);
        }

    public:

        /***********************************************************************
         *
         * Static helper functions
         *
         **********************************************************************/

        template <typename U, typename=detail::enable_if_assignable_t<len_type&,U>>
        static stride_type size(unsigned irrep, initializer_matrix<U> len)
        {
            return size<initializer_matrix<U>>(irrep, len);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        static stride_type size(unsigned irrep, std::initializer_list<U> len)
        {
            return size<std::initializer_list<U>>(irrep, len);
        }

        template <typename U, typename=detail::enable_if_container_of_containers_of_t<U,len_type>>
        static stride_type size(unsigned irrep, const U& len)
        {
            MARRAY_ASSERT(len.size() > 0);

            //TODO: add alignment option

            auto it = len.begin();
            unsigned nirrep = it->size();
            unsigned ndim = len.size();

            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            std::array<stride_type, 8> size;
            std::copy_n(it->begin(), nirrep, size.begin());
            ++it;

            for (unsigned i = 1;i < ndim;i++)
            {
                MARRAY_ASSERT(it->size() == nirrep);

                std::array<stride_type, 8> new_size = {};

                for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                {
                    auto it2 = it->begin();
                    for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                    {
                        new_size[irr1] += size[irr1^irr2]*(*it2);
                        ++it2;
                    }
                }

                size = new_size;
                ++it;
            }

            return size[irrep];
        }

        template <typename U>
        static stride_type size(unsigned irrep, matrix_view<U> len)
        {
            //TODO: add alignment option

            unsigned nirrep = len.length(1);
            unsigned ndim = len.length(0);

            MARRAY_ASSERT(len.length(0) > 0);
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            std::array<stride_type, 8> size;
            for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                size[irr1] = len[0][irr1];

            for (unsigned i = 1;i < ndim;i++)
            {
                std::array<stride_type, 8> new_size = {};

                for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                {
                    for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                    {
                        new_size[irr1] += size[irr1^irr2]*len[i][irr2];
                    }
                }

                size = new_size;
            }

            return size[irrep];
        }

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const dpd_marray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const dpd_marray_base<U, NDim, D, O>& other)
        {
            MARRAY_ASSERT(nirrep_ == other.nirrep_);
            MARRAY_ASSERT(irrep_ == other.irrep_);

            for (unsigned i = 0;i < NDim;i++)
            {
                MARRAY_ASSERT(lengths(i) == other.lengths(i));
            }

            if (layout_ == other.layout_ && perm_ == other.perm_)
            {
                std::copy_n(other.data(), size(irrep_, len_.view()), data());
            }
            else
            {
                unsigned mask = nirrep_-1;
                unsigned shift = (nirrep_>1) + (nirrep_>2) + (nirrep_>4);

                unsigned nblocks = 1u << (shift*(NDim-1));
                std::array<unsigned, NDim> irreps;
                for (unsigned block = 0;block < nblocks;block++)
                {
                    unsigned b = block;
                    irreps[0] = irrep_;
                    for (unsigned i = 1;i < NDim;i++)
                    {
                        irreps[0] ^= irreps[i] = b & mask;
                        b >>= shift;
                    }

                    (*this)(irreps) = other(irreps);
                }
            }

            return static_cast<Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            std::fill_n(data(), size(irrep_, len_.view()));
            return static_cast<Derived&>(*this);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        dpd_marray_view<const Type, NDim> cview() const
        {
            return const_cast<dpd_marray_base&>(*this).view();
        }

        dpd_marray_view<ctype, NDim> view() const
        {
            return const_cast<dpd_marray_base&>(*this).view();
        }

        dpd_marray_view<Type, NDim> view()
        {
            return *this;
        }

        friend dpd_marray_view<const Type, NDim> cview(const dpd_marray_base& x)
        {
            return x.view();
        }

        friend dpd_marray_view<ctype, NDim> view(const dpd_marray_base& x)
        {
            return x.view();
        }

        friend dpd_marray_view<Type, NDim> view(dpd_marray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        template <typename U, typename=
            detail::enable_if_assignable_t<unsigned&,U>>
        dpd_marray_view<ctype,NDim> permuted(std::initializer_list<U> perm) const
        {
            return const_cast<dpd_marray_base&>(*this).permuted(perm);
        }

        template <typename U, typename=
            detail::enable_if_assignable_t<unsigned&,U>>
        dpd_marray_view<Type,NDim> permuted(std::initializer_list<U> perm)
        {
            return permuted<std::initializer_list<U>>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        dpd_marray_view<ctype,NDim> permuted(const U& perm) const
        {
            return const_cast<dpd_marray_base&>(*this).permuted<U>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        dpd_marray_view<Type,NDim> permuted(const U& perm)
        {
            dpd_marray_view<Type,NDim> r(*this);
            r.permute(perm);
            return r;
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<ctype, NDim> transposed() const
        {
            return const_cast<dpd_marray_base&>(*this).transposed();
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<Type, NDim> transposed()
        {
            return permuted({1, 0});
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<ctype, NDim> T() const
        {
            return const_cast<dpd_marray_base&>(*this).T();
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<Type, NDim> T()
        {
            return transposed();
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        template <typename... Irreps,
            typename=detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value &&
                                         sizeof...(Irreps) == NDim>>
        marray_view<ctype, NDim> operator()(const Irreps&... irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps...);
        }

        template <typename... Irreps,
            typename=detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value &&
                                         sizeof...(Irreps) == NDim>>
        marray_view<Type, NDim> operator()(const Irreps&... irreps)
        {
            return operator()({(unsigned)irreps...});
        }

        marray_view<ctype, NDim> operator()(std::initializer_list<unsigned> irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps);
        }

        marray_view<Type, NDim> operator()(std::initializer_list<unsigned> irreps)
        {
            return operator()<>(irreps);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        marray_view<ctype, NDim> operator()(const U& irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        marray_view<Type, NDim> operator()(const U& irreps_)
        {
            MARRAY_ASSERT(irreps_.size() == NDim);

            std::array<unsigned, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            std::copy_n(irreps_.begin(), NDim, irreps.begin());

            pointer data;
            get_block(irreps, len, data, stride);

            return marray_view<Type, NDim>(len, data, stride);
        }

        template <typename U, typename V, typename W,
            typename=detail::enable_if_t<detail::is_container_of<U,unsigned>::value &&
                                         detail::is_container_of<V,len_type>::value &&
                                         detail::is_container_of<W,stride_type>::value>>
        void get_block(const U& irreps, V& len, const_pointer& data, W& stride) const
        {
            const_cast<dpd_marray_base&>(*this).get_block(irreps, len,
                                                          const_cast<pointer&>(data),
                                                          stride);
        }

        template <typename U, typename V, typename W,
            typename=detail::enable_if_t<detail::is_container_of<U,unsigned>::value &&
                                         detail::is_container_of<V,len_type>::value &&
                                         detail::is_container_of<W,stride_type>::value>>
        void get_block(const U& irreps, V& len, pointer& data, W& stride)
        {
            MARRAY_ASSERT(irreps.size() == NDim);
            MARRAY_ASSERT(len.size() == NDim);
            MARRAY_ASSERT(stride.size() == NDim);

            unsigned irrep = 0;
            for (auto& i : irreps) irrep ^= i;
            MARRAY_ASSERT(irrep == irrep_);

            data = data_;
            detail::get_block(detail::inverse_permutation(perm_), irreps,
                              len_, size_, layout_, len, data, stride);
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block<marray_view<ctype, NDim>>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block<marray_view<Type, NDim>>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<ctype>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<Type>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        /***********************************************************************
         *
         * Basic getters
         *
         **********************************************************************/

        const_pointer cdata() const
        {
            return const_cast<dpd_marray_base&>(*this).data();
        }

        cptr data() const
        {
            return const_cast<dpd_marray_base&>(*this).data();
        }

        pointer data()
        {
            return data_;
        }

        template <unsigned Dim>
        len_type length(unsigned irrep) const
        {
            static_assert(Dim < NDim, "Dim out of range");
            MARRAY_ASSERT(irrep < nirrep_);
            return len_[perm_[Dim]][irrep];
        }

        len_type length(unsigned dim, unsigned irrep) const
        {
            MARRAY_ASSERT(dim < NDim);
            MARRAY_ASSERT(irrep < nirrep_);
            return len_[perm_[dim]][irrep];
        }

        template <unsigned Dim>
        const std::array<len_type,8>& lengths() const
        {
            static_assert(Dim < NDim, "Dim out of range");
            return len_[perm_[Dim]];
        }

        const std::array<len_type,8>& lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < NDim);
            return len_[perm_[dim]];
        }

        std::array<std::array<len_type,8>, NDim> lengths() const
        {
            std::array<std::array<len_type,8>, NDim> len;
            for (unsigned i = 0;i < NDim;i++) len[i] = len_[perm_[i]];
            return len;
        }

        unsigned irrep() const
        {
            return irrep_;
        }

        unsigned num_irreps() const
        {
            return nirrep_;
        }

        const std::array<unsigned, NDim>& permutation() const
        {
            return perm_;
        }

        static constexpr unsigned dimension()
        {
            return NDim;
        }
};

}

#endif
