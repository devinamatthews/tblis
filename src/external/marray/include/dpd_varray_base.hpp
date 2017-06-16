#ifndef _MARRAY_DPD_VARRAY_BASE_HPP_
#define _MARRAY_DPD_VARRAY_BASE_HPP_

#include "dpd_marray_base.hpp"
#include "varray_view.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class dpd_varray_base
{
    template <typename, typename, bool> friend class dpd_varray_base;
    template <typename> friend class dpd_varray_view;
    template <typename, typename> friend class dpd_varray;

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
        matrix<len_type> len_;
        matrix<stride_type> size_;
        std::vector<unsigned> perm_;
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
            len_.reset();
            size_.reset();
            perm_.clear();
            data_ = nullptr;
            irrep_ = 0;
            nirrep_ = 0;
            layout_ = DEFAULT;
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::cptr,pointer>>
        void reset(const dpd_varray_base<U, D, O>& other)
        {
            reset(const_cast<dpd_varray_base<U, D, O>&>(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename dpd_varray_base<U, D, O>::pointer,pointer>>
        void reset(dpd_varray_base<U, D, O>& other)
        {
            len_.reset(other.len_);
            size_.reset(other.size_);
            perm_ = other.perm_;
            data_ = other.data_;
            irrep_ = other.irrep_;
            nirrep_ = other.nirrep_;
            layout_ = other.layout_;
        }

        void reset(unsigned irrep, unsigned nirrep,
                   initializer_matrix<len_type> len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            reset<initializer_matrix<len_type>>(irrep, nirrep, len, ptr, layout);
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

            unsigned ndim = len.size();

            MARRAY_ASSERT(ndim > 0);

            irrep_ = irrep;
            nirrep_ = nirrep;
            data_ = ptr;
            layout_ = layout;
            len_.reset({ndim, nirrep}, ROW_MAJOR);
            size_.reset({num_sizes(ndim, layout), nirrep}, ROW_MAJOR);
            perm_.resize(ndim);

            if (layout == BLOCKED_COLUMN_MAJOR ||
                layout == PREFIX_COLUMN_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                // Column major
                auto it = len.begin();
                for (unsigned i = 0;i < ndim;i++)
                {
                    auto it2 = it->begin();
                    for (unsigned j = 0;j < nirrep;j++)
                    {
                        len_[i][j] = *it2;
                        ++it2;
                    }
                    perm_[i] = i;
                    ++it;
                }
            }
            else
            {
                // Row major: reverse the dimensions and treat as
                // permuted column major
                auto it = len.begin();
                for (unsigned i = 0;i < ndim;i++)
                {
                    auto it2 = it->begin();
                    for (unsigned j = 0;j < nirrep;j++)
                    {
                        len_[ndim-1-i][j] = *it2;
                        ++it2;
                    }
                    perm_[i] = ndim-1-i;
                    ++it;
                }
            }

            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                local_size(irrep_, 0, ndim, 0);
            }
            else
            {
                size_[0] = len_[0];

                for (unsigned i = 1;i < ndim;i++)
                {
                    for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                    {
                        size_[i][irr1] = 0;
                        for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                        {
                            size_[i][irr1] += size_[i-1][irr1^irr2]*
                                              len_[i][irr2];
                        }
                    }
                }
            }
        }

        template <typename U>
        void reset(unsigned irrep, unsigned nirrep,
                   matrix_view<U> len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            unsigned ndim = len.length(0);

            MARRAY_ASSERT(ndim > 0);
            MARRAY_ASSERT(len.length(1) == nirrep);

            irrep_ = irrep;
            nirrep_ = nirrep;
            data_ = ptr;
            layout_ = layout;
            len_.reset({ndim, nirrep}, ROW_MAJOR);
            size_.reset({num_sizes(ndim, layout), nirrep}, ROW_MAJOR);
            perm_.resize(ndim);

            if (layout == BLOCKED_COLUMN_MAJOR ||
                layout == PREFIX_COLUMN_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                // Column major
                for (unsigned i = 0;i < ndim;i++)
                {
                    for (unsigned j = 0;j < nirrep;j++)
                        len_[i][j] = len[i][j];
                    perm_[i] = i;
                }
            }
            else
            {
                // Row major: reverse the dimensions and treat as
                // permuted column major
                for (unsigned i = 0;i < ndim;i++)
                {
                    for (unsigned j = 0;j < nirrep;j++)
                        len_[ndim-1-i][j] = len[i][j];
                    perm_[i] = ndim-1-i;
                }
            }

            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                local_size(irrep_, 0, ndim, 0);
            }
            else
            {
                size_[0] = len_[0];

                for (unsigned i = 1;i < ndim;i++)
                {
                    for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                    {
                        size_[i][irr1] = 0;
                        for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                        {
                            size_[i][irr1] += size_[i-1][irr1^irr2]*
                                              len_[i][irr2];
                        }
                    }
                }
            }
        }

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename View, typename Func>
        void for_each_block(Func&& f) const
        {
            typedef typename View::pointer Ptr;

            unsigned ndim = dimension();

            const_pointer cptr;
            std::vector<unsigned> irreps(ndim);
            std::vector<len_type> len(ndim);
            std::vector<stride_type> stride(ndim);

            viterator<0> it(std::vector<unsigned>(ndim-1, nirrep_));
            while (it.next())
            {
                irreps[0] = irrep_;
                for (unsigned i = 1;i < ndim;i++)
                {
                    irreps[0] ^= irreps[i] = it.position()[i-1];
                }

                get_block(irreps, len, cptr, stride);

                f(View(len, const_cast<Ptr>(cptr), stride), irreps);
            }
        }

        template <typename View, typename Func, unsigned... I>
        void for_each_block(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            constexpr unsigned NDim = sizeof...(I);
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
            constexpr unsigned NDim = sizeof...(I);
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

        void local_size(unsigned irrep, unsigned begin, unsigned end,
                        unsigned idx)
        {
            unsigned ndim = end-begin;

            if (ndim == 1)
            {
                size_[idx][irrep] = len_[begin][irrep];
                return;
            }

            unsigned mid = begin+(ndim+1)/2;
            unsigned idxl = idx+1;
            unsigned idxr = idx+((ndim+1)/2)*2;

            stride_type size = 0;
            for (unsigned irrepr = 0;irrepr < nirrep_;irrepr++)
            {
                unsigned irrepl = irrepr^irrep;
                local_size(irrepl, begin, mid, idxl);
                local_size(irrepr,   mid, end, idxr);
                size += size_[idxl][irrepl]*size_[idxr][irrepr];
            }

            size_[idx][irrep] = size;
        }

        template <typename U, typename V>
        unsigned local_offset(const std::vector<unsigned>& iperm,
                              const U& irreps,
                              stride_type size_before, pointer& data,
                              V& stride,
                              unsigned begin, unsigned end, unsigned idx)
        {
            unsigned ndim = end-begin;

            if (ndim == 1)
            {
                stride[iperm[begin]] = size_before;
                return irreps[iperm[begin]];
            }

            unsigned mid = begin+(ndim+1)/2;
            unsigned idxl = idx+1;
            unsigned idxr = idx+((ndim+1)/2)*2;

            unsigned irrepl = local_offset(iperm, irreps, size_before,
                                           data, stride, begin, mid, idxl);
            unsigned irrepr = local_offset(iperm, irreps, size_before*size_[idxl][irrepl],
                                           data, stride, mid, end, idxr);
            unsigned irrep = irrepl^irrepr;

            for (unsigned irr2 = 0;irr2 < irrepr;irr2++)
            {
                unsigned irr1 = irr2^irrep;
                data += size_before*size_[idxl][irr1]*size_[idxr][irr2];
            }

            return irrep;
        }

        static unsigned num_sizes(unsigned ndim, dpd_layout layout)
        {
            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR) return 2*ndim-1;
            else return ndim;
        }

        void swap(dpd_varray_base& other)
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

        static stride_type size(unsigned irrep, initializer_matrix<len_type> len)
        {
            return size<>(irrep, len);
        }

        template <typename U, typename=detail::enable_if_container_of_containers_of_t<U,len_type>>
        static stride_type size(unsigned irrep, const U& len)
        {
            return dpd_marray_base<Type,1,dpd_marray_view<Type,1>,false>::size(irrep, len);
        }

        template <typename U>
        static stride_type size(unsigned irrep, matrix_view<U> len)
        {
            return dpd_marray_base<Type,1,dpd_marray_view<Type,1>,false>::size(irrep, len);
        }

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const dpd_varray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_t<std::is_assignable<reference,U>::value>>
        Derived& operator=(const dpd_varray_base<U, D, O>& other)
        {
            unsigned ndim = dimension();

            MARRAY_ASSERT(ndim == other.dimension());
            MARRAY_ASSERT(nirrep_ == other.nirrep_);
            MARRAY_ASSERT(irrep_ == other.irrep_);

            for (unsigned i = 0;i < ndim;i++)
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

                unsigned nblocks = 1u << (shift*(ndim-1));
                std::vector<unsigned> irreps(ndim);
                for (unsigned block = 0;block < nblocks;block++)
                {
                    unsigned b = block;
                    irreps[0] = irrep_;
                    for (unsigned i = 1;i < ndim;i++)
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

        dpd_varray_view<const Type> cview() const
        {
            return const_cast<dpd_varray_base&>(*this).view();
        }

        dpd_varray_view<ctype> view() const
        {
            return const_cast<dpd_varray_base&>(*this).view();
        }

        dpd_varray_view<Type> view()
        {
            return *this;
        }

        friend dpd_varray_view<const Type> cview(const dpd_varray_base& x)
        {
            return x.view();
        }

        friend dpd_varray_view<ctype> view(const dpd_varray_base& x)
        {
            return x.view();
        }

        friend dpd_varray_view<Type> view(dpd_varray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        dpd_varray_view<ctype> permuted(std::initializer_list<unsigned> perm) const
        {
            return const_cast<dpd_varray_base&>(*this).permuted(perm);
        }

        dpd_varray_view<Type> permuted(std::initializer_list<unsigned> perm)
        {
            return permuted<>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        dpd_varray_view<ctype> permuted(const U& perm) const
        {
            return const_cast<dpd_varray_base&>(*this).permuted<U>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        dpd_varray_view<Type> permuted(const U& perm)
        {
            dpd_varray_view<Type> r(*this);
            r.permute(perm);
            return r;
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        template <typename... Irreps,
            typename=detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value>>
        marray_view<ctype, sizeof...(Irreps)> operator()(const Irreps&... irreps) const
        {
            return const_cast<dpd_varray_base&>(*this)(irreps...);
        }

        template <typename... Irreps,
            typename=detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value>>
        marray_view<Type, sizeof...(Irreps)> operator()(const Irreps&... irreps_)
        {
            constexpr unsigned NDim = sizeof...(Irreps);

            MARRAY_ASSERT(NDim == dimension());

            std::array<unsigned, NDim> irreps{(unsigned)irreps_...};
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            pointer data;
            get_block(irreps, len, data, stride);

            return marray_view<Type, NDim>(len, data, stride);
        }

        varray_view<ctype> operator()(std::initializer_list<unsigned> irreps) const
        {
            return const_cast<dpd_varray_base&>(*this)(irreps);
        }

        varray_view<Type> operator()(std::initializer_list<unsigned> irreps)
        {
            return operator()<>(irreps);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<ctype> operator()(const U& irreps) const
        {
            return const_cast<dpd_varray_base&>(*this)(irreps);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        varray_view<Type> operator()(const U& irreps_)
        {
            unsigned ndim = dimension();

            std::vector<unsigned> irreps(irreps_.begin(), irreps_.end());
            std::vector<len_type> len(ndim);
            std::vector<stride_type> stride(ndim);

            pointer data;
            get_block(irreps, len, data, stride);

            return varray_view<Type>(len, data, stride);
        }

        template <typename U, typename V, typename W,
            typename=detail::enable_if_t<detail::is_container_of<U,unsigned>::value &&
                                         detail::is_container_of<V,len_type>::value &&
                                         detail::is_container_of<W,stride_type>::value>>
        void get_block(const U& irreps, V& len, const_pointer& data, W& stride) const
        {
            const_cast<dpd_varray_base&>(*this).get_block(irreps, len,
                                                          const_cast<pointer&>(data),
                                                          stride);
        }

        template <typename U, typename V, typename W,
            typename=detail::enable_if_t<detail::is_container_of<U,unsigned>::value &&
                                         detail::is_container_of<V,len_type>::value &&
                                         detail::is_container_of<W,stride_type>::value>>
        void get_block(const U& irreps, V& len, pointer& data, W& stride)
        {
            unsigned ndim = dimension();

            MARRAY_ASSERT(irreps.size() == ndim);
            MARRAY_ASSERT(len.size() == ndim);
            MARRAY_ASSERT(stride.size() == ndim);

            unsigned irrep = 0;
            for (auto& i : irreps) irrep ^= i;
            MARRAY_ASSERT(irrep == irrep_);

            auto iperm = detail::inverse_permutation(perm_);

            data = data_;

            for (unsigned i = 0;i < ndim;i++)
                len[i] = len_[perm_[i]][irreps[i]];

            if (layout_ == BALANCED_ROW_MAJOR ||
                layout_ == BALANCED_COLUMN_MAJOR)
            {
                local_offset(iperm, irreps, 1, data, stride, 0, ndim, 0);
            }
            else
            {
                if (layout_ == PREFIX_ROW_MAJOR ||
                    layout_ == PREFIX_COLUMN_MAJOR)
                {
                    unsigned lirrep = 0;
                    stride[iperm[0]] = 1;
                    for (unsigned i = 1;i < ndim;i++)
                    {
                        lirrep ^= irreps[iperm[i-1]];
                        stride[iperm[i]] = size_[i-1][lirrep];
                    }

                    unsigned rirrep = irrep_;
                    for (unsigned i = ndim;i --> 1;)
                    {
                        for (unsigned irr1 = 0;irr1 < irreps[iperm[i]];irr1++)
                        {
                            unsigned irr2 = irr1^rirrep;
                            data += size_[i-1][irr2]*len_[i][irr1];
                        }
                        rirrep ^= irreps[iperm[i]];
                    }
                }
                else
                {
                    stride_type lsize = 1;
                    for (unsigned i = 0;i < ndim;i++)
                    {
                        stride[iperm[i]] = lsize;
                        lsize *= len[iperm[i]];
                    }

                    stride_type rsize = 1;
                    unsigned rirrep = irrep_;
                    for (unsigned i = ndim;i --> 1;)
                    {
                        stride_type offset = 0;
                        for (unsigned irr1 = 0;irr1 < irreps[iperm[i]];irr1++)
                        {
                            unsigned irr2 = irr1^rirrep;
                            offset += size_[i-1][irr2]*len_[i][irr1];
                        }
                        data += offset*rsize;
                        rsize *= len[iperm[i]];
                        rirrep ^= irreps[iperm[i]];
                    }
                }
            }
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block<varray_view<ctype>>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block<varray_view<Type>>(std::forward<Func>(f));
        }

        template <unsigned NDim, typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block<marray_view<ctype, NDim>>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <unsigned NDim, typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block<marray_view<Type, NDim>>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_block(
            [&](const varray_view<ctype>& view, const std::vector<unsigned>& irreps)
            {
                view.for_each_element(
                [&](cref value, const std::vector<len_type>& pos)
                {
                    f(value, irreps, pos);
                });
            });
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_block(
            [&](const varray_view<Type>& view, const std::vector<unsigned>& irreps)
            {
                view.for_each_element(
                [&](reference value, const std::vector<len_type>& pos)
                {
                    f(value, irreps, pos);
                });
            });
        }

        template <unsigned NDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<ctype>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <unsigned NDim, typename Func>
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
            return const_cast<dpd_varray_base&>(*this).data();
        }

        cptr data() const
        {
            return const_cast<dpd_varray_base&>(*this).data();
        }

        pointer data()
        {
            return data_;
        }

        len_type length(unsigned dim, unsigned irrep) const
        {
            MARRAY_ASSERT(dim < dimension());
            MARRAY_ASSERT(irrep < nirrep_);
            return len_[perm_[dim]][irrep];
        }

        row_view<const len_type> lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < dimension());
            return len_[perm_[dim]];
        }

        matrix<len_type> lengths() const
        {
            unsigned ndim = dimension();
            matrix<len_type> len({ndim, nirrep_}, ROW_MAJOR);
            for (unsigned i = 0;i < ndim;i++) len[i] = len_[perm_[i]];
            return len;
        }

        unsigned irrep() const
        {
            return irrep_;
        }

        unsigned num_irreps() const
        {
            return len_.length(1);
        }

        const std::vector<unsigned>& permutation() const
        {
            return perm_;
        }

        unsigned dimension() const
        {
            return len_.length(0);
        }
};

}

#endif
