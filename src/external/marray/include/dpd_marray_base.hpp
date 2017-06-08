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

    protected:
        typedef typename std::conditional<Owner,const Type,Type>::type ctype;
        typedef ctype& cref;
        typedef ctype* cptr;
        template <typename U> using initializer_matrix =
            std::initializer_list<std::initializer_list<U>>;

        std::array<std::array<len_type,8>, NDim> len_ = {};
        std::array<std::array<stride_type,8>, 2*NDim> size_ = {};
        std::array<unsigned, NDim> perm_ = {};
        pointer data_ = nullptr;
        unsigned irrep_ = 0;
        unsigned nirrep_ = 0;
        dpd_layout layout_ = DEFAULT;

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

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

        unsigned local_offset(const std::array<unsigned, NDim>& irreps,
                              stride_type size_before, pointer& data,
                              std::array<stride_type, NDim>& stride,
                              unsigned begin, unsigned end, unsigned idx)
        {
            unsigned ndim = end-begin;

            if (ndim == 1)
            {
                stride[begin] = size_before;
                return irreps[begin];
            }

            unsigned mid = begin+(ndim+1)/2;
            unsigned idxl = idx+1;
            unsigned idxr = idx+((ndim+1)/2)*2;

            unsigned irrepl = local_offset(irreps, size_before,
                                           data, stride, begin, mid, idxl);
            unsigned irrepr = local_offset(irreps, size_before*size_[idxl][irrepl],
                                           data, stride, mid, end, idxr);
            unsigned irrep = irrepl^irrepr;

            for (unsigned irr2 = 0;irr2 < irrepr;irr2++)
            {
                unsigned irr1 = irr2^irrep;
                data += size_before*size_[idxl][irr1]*size_[idxr][irr2];
            }

            return irrep;
        }

        static unsigned num_sizes(dpd_layout layout)
        {
            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR) return 2*NDim-1;
            else return NDim;
        }

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

            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                local_size(irrep_, 0, NDim, 0);
            }
            else
            {
                size_[0] = len_[0];

                for (unsigned i = 1;i < NDim;i++)
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

            if (layout == BALANCED_ROW_MAJOR ||
                layout == BALANCED_COLUMN_MAJOR)
            {
                local_size(irrep_, 0, NDim, 0);
            }
            else
            {
                size_[0] = len_[0];

                for (unsigned i = 1;i < NDim;i++)
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

        static stride_type size(unsigned irrep, initializer_matrix<len_type> len)
        {
            return size<>(irrep, len);
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

        dpd_marray_view<ctype,NDim> permuted(std::initializer_list<unsigned> perm) const
        {
            return const_cast<dpd_marray_base&>(*this).permuted(perm);
        }

        dpd_marray_view<Type,NDim> permuted(std::initializer_list<unsigned> perm)
        {
            return permuted<>(perm);
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
        marray_view<Type, NDim> operator()(const U& irreps)
        {
            unsigned irrep = 0;
            auto it = irreps.begin();
            std::array<unsigned, NDim> irreps_;
            for (unsigned i = 0;i < NDim;i++)
            {
                irrep ^= irreps_[perm_[i]] = *it;
                ++it;
            }
            MARRAY_ASSERT(irrep == irrep_);

            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;
            pointer data = data_;

            for (unsigned i = 0;i < NDim;i++)
                len[i] = len_[i][irreps_[i]];

            if (layout_ == BALANCED_ROW_MAJOR ||
                layout_ == BALANCED_COLUMN_MAJOR)
            {
                local_offset(irreps_, 1, data, stride, 0, NDim, 0);
            }
            else
            {
                if (layout_ == PREFIX_ROW_MAJOR ||
                    layout_ == PREFIX_COLUMN_MAJOR)
                {
                    unsigned lirrep = 0;
                    stride[0] = 1;
                    for (unsigned i = 1;i < NDim;i++)
                    {
                        lirrep ^= irreps_[i-1];
                        stride[i] = size_[i-1][lirrep];
                    }

                    unsigned rirrep = irrep_;
                    for (unsigned i = NDim;i --> 1;)
                    {
                        for (unsigned irr1 = 0;irr1 < irreps_[i];irr1++)
                        {
                            unsigned irr2 = irr1^rirrep;
                            data += size_[i-1][irr2]*len_[i][irr1];
                        }
                        rirrep ^= irreps_[i];
                    }
                }
                else
                {
                    stride_type lsize = 1;
                    for (unsigned i = 0;i < NDim;i++)
                    {
                        stride[i] = lsize;
                        lsize *= len[i];
                    }

                    stride_type rsize = 1;
                    unsigned rirrep = irrep_;
                    for (unsigned i = NDim;i --> 1;)
                    {
                        stride_type offset = 0;
                        for (unsigned irr1 = 0;irr1 < irreps_[i];irr1++)
                        {
                            unsigned irr2 = irr1^rirrep;
                            offset += size_[i-1][irr2]*len_[i][irr1];
                        }
                        data += offset*rsize;
                        rsize *= len[i];
                        rirrep ^= irreps_[i];
                    }
                }
            }

            return marray_view<Type, NDim>(len, data, stride).permuted(perm_);
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
