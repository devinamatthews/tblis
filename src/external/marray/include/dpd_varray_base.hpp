#ifndef _MARRAY_DPD_VARRAY_BASE_HPP_
#define _MARRAY_DPD_VARRAY_BASE_HPP_

#include "dpd_marray_base.hpp"
#include "varray_view.hpp"

namespace MArray
{

template <typename Type, typename Derived, bool Owner>
class dpd_varray_base : protected detail::dpd_base<dpd_varray_base<Type, Derived, Owner>>
{
    template <typename> friend struct detail::dpd_base;
    template <typename, typename, bool> friend class dpd_varray_base;
    template <typename> friend class dpd_varray_view;
    template <typename, typename> friend class dpd_varray;
    template <typename, typename, bool> friend class indexed_dpd_varray_base;

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
        matrix<stride_type> size_;
        matrix<len_type> len_;
        matrix<len_type> off_;
        matrix<stride_type> stride_;
        dim_vector leaf_;
        dim_vector parent_;
        dim_vector perm_;
        dim_vector depth_;
        pointer data_ = nullptr;
        unsigned irrep_ = 0;
        unsigned nirrep_ = 0;
        layout layout_ = DEFAULT;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            size_.reset();
            len_.reset();
            off_.reset();
            stride_.reset();
            leaf_.clear();
            parent_.clear();
            perm_.clear();
            depth_.clear();
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
            size_.reset(other.size_);
            len_.reset(other.len_);
            off_.reset(other.off_);
            stride_.reset(other.stride_);
            leaf_ = other.leaf_;
            parent_ = other.parent_;
            perm_ = other.perm_;
            depth_ = other.depth_;
            data_ = other.data_;
            irrep_ = other.irrep_;
            nirrep_ = other.nirrep_;
            layout_ = other.layout_;
        }

        void reset(unsigned irrep, unsigned nirrep,
                   const detail::array_2d<len_type>& len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr,
                  this->default_depth(layout, len.length(0)), layout.base());
        }

        void reset(unsigned irrep, unsigned nirrep,
                   const detail::array_2d<len_type>& len, pointer ptr,
                   const detail::array_1d<unsigned>& depth, layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            unsigned ndim = len.length(0);
            MARRAY_ASSERT(ndim > 0);
            MARRAY_ASSERT(len.length(1) >= nirrep);
            MARRAY_ASSERT(depth.size() == ndim);

            data_ = ptr;
            irrep_ = irrep;
            nirrep_ = nirrep;
            layout_ = layout;
            size_.reset({2*ndim-1, nirrep}, ROW_MAJOR);
            len.slurp(len_, ROW_MAJOR);
            off_.reset({ndim, nirrep}, ROW_MAJOR);
            stride_.reset({ndim, nirrep}, 1, ROW_MAJOR);
            leaf_.resize(ndim);
            parent_.resize(2*ndim-1);
            perm_.resize(ndim);
            depth.slurp(depth_);

            this->set_tree();
            this->set_size();
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
            irrep_vector irreps(ndim);
            len_vector len(ndim);
            stride_vector stride(ndim);

            viterator<0> it(irrep_vector(ndim-1, nirrep_));
            while (it.next())
            {
                irreps[0] = irrep_;
                for (unsigned i = 1;i < ndim;i++)
                {
                    irreps[0] ^= irreps[i] = it.position()[i-1];
                }

                bool empty = false;
                for (unsigned i = 0;i < ndim;i++)
                {
                    if (!length(i, irreps[i])) empty = true;
                }
                if (empty) continue;

                cptr = data();
                this->get_block(irreps, len, cptr, stride);

                detail::call(std::forward<Func>(f),
                             View(len, const_cast<Ptr>(cptr), stride),
                             irreps);
            }
        }

        template <typename View, typename Func, unsigned... I>
        void for_each_block(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            constexpr unsigned NDim = sizeof...(I);
            typedef typename View::pointer Ptr;

            MARRAY_ASSERT(NDim == dimension());

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

                bool empty = false;
                for (unsigned i = 0;i < NDim;i++)
                {
                    if (!length(i, irreps[i])) empty = true;
                }
                if (empty) continue;

                cptr = data();
                this->get_block(irreps, len, cptr, stride);

                detail::call(std::forward<Func>(f),
                             View(len, const_cast<Ptr>(cptr), stride),
                             irreps[I]...);
            }
        }

        template <typename Tp, typename Func>
        void for_each_element(Func&& f) const
        {
            typedef Tp* Ptr;

            unsigned ndim = dimension();

            const_pointer cptr = data_;
            irrep_vector irreps(ndim);
            len_vector len(ndim);
            stride_vector stride(ndim);

            viterator<0> it1(irrep_vector(ndim-1, nirrep_));
            while (it1.next())
            {
                irreps[0] = irrep_;
                for (unsigned i = 1;i < ndim;i++)
                {
                    irreps[0] ^= irreps[i] = it1.position()[i-1];
                }

                bool empty = false;
                for (unsigned i = 0;i < ndim;i++)
                {
                    if (!length(i, irreps[i])) empty = true;
                }
                if (empty) continue;

                cptr = data();
                this->get_block(irreps, len, cptr, stride);

                viterator<1> it2(len, stride);
                Ptr ptr = const_cast<Ptr>(cptr);
                while (it2.next(ptr)) detail::call(std::forward<Func>(f), *ptr,
                                                   irreps, it2.position());
            }
        }

        template <typename Tp, typename Func, unsigned... I>
        void for_each_element(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            constexpr unsigned NDim = sizeof...(I);
            typedef Tp* Ptr;

            MARRAY_ASSERT(NDim == dimension());

            std::array<unsigned, NDim-1> nirrep;
            nirrep.fill(nirrep_);

            const_pointer cptr = data_;
            std::array<unsigned, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            miterator<NDim-1, 0> it1(nirrep);
            while (it1.next())
            {
                irreps[0] = irrep_;
                for (unsigned i = 1;i < NDim;i++)
                {
                    irreps[0] ^= irreps[i] = it1.position()[i-1];
                }

                bool empty = false;
                for (unsigned i = 0;i < NDim;i++)
                {
                    if (!length(i, irreps[i])) empty = true;
                }
                if (empty) continue;

                cptr = data();
                this->get_block(irreps, len, cptr, stride);

                miterator<NDim, 1> it2(len, stride);
                Ptr ptr = const_cast<Ptr>(cptr);
                while (it2.next(ptr)) detail::call(std::forward<Func>(f), *ptr,
                                                   irreps[I]..., it2.position()[I]...);
            }
        }

        void swap(dpd_varray_base& other)
        {
            using std::swap;
            swap(size_, other.size_);
            swap(len_, other.len_);
            swap(off_, other.off_);
            swap(stride_, other.stride_);
            swap(leaf_, other.leaf_);
            swap(parent_, other.parent_);
            swap(perm_, other.perm_);
            swap(depth_, other.depth_);
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

        static stride_type size(unsigned irrep, const detail::array_2d<len_type>& len)
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

            unsigned mask = nirrep_-1;
            unsigned shift = (nirrep_>1) + (nirrep_>2) + (nirrep_>4);

            unsigned nblocks = 1u << (shift*(ndim-1));
            irrep_vector irreps(ndim);
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

            return static_cast<Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            unsigned ndim = dimension();
            unsigned mask = nirrep_-1;
            unsigned shift = (nirrep_>1) + (nirrep_>2) + (nirrep_>4);

            unsigned nblocks = 1u << (shift*(ndim-1));
            irrep_vector irreps(ndim);
            for (unsigned block = 0;block < nblocks;block++)
            {
                unsigned b = block;
                irreps[0] = irrep_;
                for (unsigned i = 1;i < ndim;i++)
                {
                    irreps[0] ^= irreps[i] = b & mask;
                    b >>= shift;
                }

                (*this)(irreps) = value;
            }

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

        dpd_varray_view<ctype> permuted(const detail::array_1d<unsigned>& perm) const
        {
            return const_cast<dpd_varray_base&>(*this).permuted(perm);
        }

        dpd_varray_view<Type> permuted(const detail::array_1d<unsigned>& perm)
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

        template <typename... Irreps>
        detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value,
                            marray_view<ctype, sizeof...(Irreps)>>
        operator()(const Irreps&... irreps) const
        {
            return const_cast<dpd_varray_base&>(*this)(irreps...);
        }

        template <typename... Irreps>
        detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value,
                            marray_view<Type, sizeof...(Irreps)>>
        operator()(const Irreps&... irreps_)
        {
            constexpr unsigned NDim = sizeof...(Irreps);

            MARRAY_ASSERT(NDim == dimension());

            std::array<unsigned, NDim> irreps{(unsigned)irreps_...};
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            pointer ptr = data();
            this->get_block(irreps, len, ptr, stride);

            return marray_view<Type, NDim>(len, ptr, stride);
        }

        varray_view<ctype> operator()(const detail::array_1d<unsigned>& irreps) const
        {
            return const_cast<dpd_varray_base&>(*this)(irreps);
        }

        varray_view<Type> operator()(const detail::array_1d<unsigned>& irreps_)
        {
            unsigned ndim = dimension();

            MARRAY_ASSERT(irreps_.size() == ndim);

            irrep_vector irreps;
            irreps_.slurp(irreps);

            len_vector len(ndim);
            stride_vector stride(ndim);

            unsigned irrep = 0;
            for (auto& i: irreps) irrep ^= i;
            MARRAY_ASSERT(irrep == irrep_);

            pointer ptr = data();
            this->get_block(irreps, len, ptr, stride);

            return varray_view<Type>(len, ptr, stride);
        }

        /***********************************************************************
         *
         * Slicing
         *
         **********************************************************************/

        template <typename... Slices>
        detail::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value &&
                            (detail::sliced_dimension<Slices...>::value > 0),
                            dpd_marray_view<ctype, detail::sliced_dimension<Slices...>::value>>
        operator()(const Slices&... slices) const
        {
            return const_cast<dpd_varray_base&>(*this)(slices...);
        }

        template <typename... Slices>
        detail::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value &&
                            (detail::sliced_dimension<Slices...>::value > 0),
                            dpd_marray_view<Type, detail::sliced_dimension<Slices...>::value>>
        operator()(const Slices&... slices)
        {
            constexpr unsigned NDim = detail::sliced_dimension<Slices...>::value;

            abort();
            //TODO
        }

        dpd_varray_view<ctype> operator()(const detail::array_1d<dpd_range_t>& slices) const
        {
            return const_cast<dpd_varray_base&>(*this)(slices);
        }

        dpd_varray_view<Type> operator()(const detail::array_1d<dpd_range_t>& slices_)
        {
            unsigned ndim = dimension();

            MARRAY_ASSERT(slices_.size() == ndim);

            short_vector<dpd_range_t, MARRAY_OPT_NDIM> slices;
            slices_.slurp(slices);

            dpd_varray_view<Type> v = view();

            for (unsigned i = 0;i < ndim;i++)
            {
                for (unsigned j = 0;j < nirrep_;j++)
                {
                    v.off_[i][j] = slices[i].from(j);
                    v.len_[i][j] = slices[i].size(j);
                }
            }

            return v;
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
            for_each_element<ctype>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<Type>(std::forward<Func>(f));
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
            for (unsigned i = 0;i < ndim;i++) len[i] = lengths(i);
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

        const dim_vector& permutation() const
        {
            return perm_;
        }

        unsigned dimension() const
        {
            return perm_.size();
        }
};

}

#endif
