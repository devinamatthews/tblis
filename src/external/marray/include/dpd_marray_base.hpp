#ifndef _MARRAY_DPD_MARRAY_BASE_HPP_
#define _MARRAY_DPD_MARRAY_BASE_HPP_

#include "marray_view.hpp"
#include "dpd_range.hpp"

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

template <typename Derived>
struct dpd_base
{
    static dim_vector default_depth(dpd_layout layout, len_type ndim)
    {
        dim_vector depth(ndim);

        if (layout == BALANCED_ROW_MAJOR ||
            layout == BALANCED_COLUMN_MAJOR)
        {
            unsigned dl = sizeof(unsigned)*8 - __builtin_clz(ndim) - 1;
            unsigned du = dl + (ndim&(ndim-1) ? 1 : 0);
            unsigned o = ndim - (1<<dl);
            unsigned k = 2*((o+1)/2);
            unsigned l = 2*(o/2);
            unsigned h = (ndim+1)/2;

            for (unsigned i = 0;i < k;i++) depth[i] = du;
            for (unsigned i = k;i < h;i++) depth[i] = dl;
            for (unsigned i = h;i < h+l;i++) depth[i] = du;
            for (unsigned i = h+l;i < ndim;i++) depth[i] = dl;
        }
        else if (layout == PREFIX_ROW_MAJOR ||
                 layout == PREFIX_COLUMN_MAJOR)
        {
            depth[0] = ndim-1;
            for (unsigned i = 1;i < ndim;i++) depth[i] = ndim-i;
        }
        else //if (layout == BLOCKED_ROW_MAJOR ||
             //    layout == BLOCKED_COLUMN_MAJOR)
        {
            for (unsigned i = 0;i < ndim-1;i++) depth[i] = i+1;
            depth[ndim-1] = ndim-1;
        }

        return depth;
    }

    void set_tree()
    {
        auto& leaf = derived().leaf_;
        auto& parent = derived().parent_;
        auto nirrep = derived().nirrep_;
        unsigned ndim = leaf.size();
        dim_vector depth(derived().depth_.begin(), derived().depth_.end());
        dim_vector node(ndim);
        len_vector leaf_idx = range(ndim);

        MARRAY_ASSERT(depth.size() == ndim);
        for (unsigned i = 0;i < depth.size();i++)
            MARRAY_ASSERT(depth[i] < ndim);

        unsigned pos = 0;
        for (unsigned d = ndim;d --> 0;)
        {
            for (unsigned i = 0;i < depth.size();i++)
            {
                /*
                 * If we encounter a pair of nodes with depth greater than the
                 * current level, combine them into an interior node and
                 * assign the parent pointers.
                 */
                if (depth[i] == d+1)
                {
                    MARRAY_ASSERT(i < depth.size()-1 && depth[i+1] == d+1);
                    parent[node[i]] = parent[node[i+1]] = pos;
                    depth.erase(depth.begin()+i+1);
                    depth[i]--;
                    node.erase(node.begin()+i+1);
                    node[i] = pos++;
                    leaf_idx.erase(leaf_idx.begin()+i+1);
                    leaf_idx[i] = -1;
                }
                /*
                 * For nodes on the current depth level, assign a position
                 * in the linearized tree.
                 */
                else if (depth[i] == d)
                {
                    node[i] = pos++;
                    if (leaf_idx[i] != -1) leaf[leaf_idx[i]] = node[i];
                }
            }
        }

        MARRAY_ASSERT(pos == 2*ndim-1);
        MARRAY_ASSERT(depth.size() == 1);
        MARRAY_ASSERT(depth[0] == 0);
    }

    void set_size()
    {
        auto& perm = derived().perm_;
        auto& size = derived().size_;
        auto& len = derived().len_;
        const auto& leaf = derived().leaf_;
        const auto& parent = derived().parent_;
        auto nirrep = derived().nirrep_;
        auto layout = derived().layout_;
        unsigned ndim = perm.size();

        if (layout == COLUMN_MAJOR)
        {
            // Column major
            for (unsigned i = 0;i < ndim;i++)
            {
                std::copy_n(&len[i][0], nirrep, &size[leaf[i]][0]);
                perm[i] = i;
            }
        }
        else
        {
            // Row major: reverse the dimensions and treat as
            // permuted column major
            for (unsigned i = 0;i < ndim;i++)
            {
                std::copy_n(&len[i][0], nirrep, &size[leaf[ndim-1-i]][0]);
                perm[i] = ndim-1-i;
            }

            for (unsigned i = 0;i < ndim/2;i++)
                for (unsigned j = 0;j < nirrep;j++)
                    std::swap(len[i][j], len[ndim-1-i][j]);
        }

        for (unsigned i = 0;i < ndim-1;i++)
        {
            unsigned next = parent[2*i];

            for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
            {
                size[next][irr1] = 0;
                for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                {
                    size[next][irr1] += size[2*i][irr1^irr2]*size[2*i+1][irr2];
                }
            }
        }
    }

    template <typename U, typename V, typename W, typename X>
    void get_block(const U& irreps, V& len, W& data, X& stride) const
    {
        auto& perm = derived().perm_;
        auto& size = derived().size_;
        auto& extent = derived().len_;
        auto& off = derived().off_;
        auto& leap = derived().stride_;
        auto& leaf = derived().leaf_;
        auto& parent = derived().parent_;
        unsigned ndim = perm.size();

        short_vector<unsigned, 2*MARRAY_OPT_NDIM-1> dpd_irrep(2*ndim-1);
        short_vector<stride_type, 2*MARRAY_OPT_NDIM-1> dpd_stride(2*ndim-1);
        dpd_stride[2*ndim-2] = 1;

        auto it = irreps.begin();
        for (unsigned i = 0;i < ndim;i++)
        {
            dpd_irrep[leaf[perm[i]]] = *it;
            ++it;
        }

        for (unsigned i = 0;i < ndim-1;i++)
        {
            dpd_irrep[parent[2*i]] = dpd_irrep[2*i]^dpd_irrep[2*i+1];
        }

        for (unsigned i = ndim-1;i --> 0;)
        {
            unsigned irrep = dpd_irrep[parent[2*i]];

            dpd_stride[2*i] = dpd_stride[parent[2*i]];
            dpd_stride[2*i+1] = dpd_stride[2*i]*size[2*i][dpd_irrep[2*i]];

            stride_type offset = 0;
            for (unsigned irr1 = 0;irr1 < dpd_irrep[2*i+1];irr1++)
            {
                offset += size[2*i][irr1^irrep]*size[2*i+1][irr1];
            }

            data += offset*dpd_stride[2*i];
        }

        it = irreps.begin();
        auto l = len.begin();
        auto s = stride.begin();
        for (unsigned i = 0;i < ndim;i++)
        {
            auto stride = dpd_stride[leaf[perm[i]]]*leap[perm[i]][dpd_irrep[leaf[perm[i]]]];
            *s = stride;
            *l = extent[perm[i]][*it];
            data += stride*off[perm[i]][*it];
            ++it;
            ++l;
            ++s;
        }
    }

    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    Derived& derived() { return static_cast<Derived&>(*this); }
};

}

template <typename Type, unsigned NDim, typename Derived, bool Owner>
class dpd_marray_base : protected detail::dpd_base<dpd_marray_base<Type, NDim, Derived, Owner>>
{
    static_assert(NDim > 0, "NDim must be positive");

    template <typename> friend struct detail::dpd_base;
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

    protected:
        std::array<std::array<stride_type,8>, 2*NDim-1> size_ = {};
        std::array<std::array<len_type,8>, NDim> len_ = {};
        std::array<std::array<len_type,8>, NDim> off_ = {};
        std::array<std::array<stride_type,8>, NDim> stride_ = {};
        std::array<unsigned, NDim> leaf_ = {};
        std::array<unsigned, 2*NDim-1> parent_ = {};
        std::array<unsigned, NDim> perm_ = {};
        std::array<unsigned, NDim> depth_ = {};
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
            size_ = {};
            len_ = {};
            off_ = {};
            stride_ = {};
            leaf_ = {};
            parent_ = {};
            perm_ = {};
            depth_ = {};
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
            size_ = other.size_;
            len_ = other.len_;
            off_ = other.off_;
            stride_ = other.stride_;
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
                  this->default_depth(layout, NDim), layout.base());
        }
        
        void reset(unsigned irrep, unsigned nirrep,
                   const detail::array_2d<len_type>& len, pointer ptr,
                   const detail::array_1d<unsigned>& depth, layout layout = DEFAULT)
        {
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            MARRAY_ASSERT(len.length(0) == NDim);
            MARRAY_ASSERT(len.length(1) >= nirrep);
            MARRAY_ASSERT(depth.size() == NDim);

            data_ = ptr;
            irrep_ = irrep;
            nirrep_ = nirrep;
            layout_ = layout;
            depth.slurp(depth_);
            len.slurp(len_);
            std::fill_n(&stride_[0][0], NDim*8, 1);

            this->set_tree();
            this->set_size();
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

                bool empty = false;
                for (unsigned i = 0;i < NDim;i++)
                {
                    if (length(i, irreps[i]) == 0) empty = true;
                }
                if (empty) continue;

                cptr = data();
                this->get_block(irreps, len, cptr, stride);

                detail::call(std::forward<Func>(f),
                             View(len, const_cast<Ptr>(cptr), stride),
                             irreps[I]...);
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
                    if (length(i, irreps[i]) == 0) empty = true;
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

        void swap(dpd_marray_base& other)
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

        static stride_type size(unsigned irrep, const detail::array_2d<len_type>& len_)
        {
            if (len_.length(0) == 0) return 1;

            //TODO: add alignment option

            matrix<len_type> len;
            len_.slurp(len);

            unsigned nirrep = len.length(1);
            unsigned ndim = len.length(0);

            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            std::array<stride_type, 8> size;

            for (unsigned irr = 0;irr < nirrep;irr++)
                size[irr] = len[0][irr];

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

        Derived& operator=(const Type& value)
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

                (*this)(irreps) = value;
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

        dpd_marray_view<ctype,NDim> permuted(const detail::array_1d<unsigned>& perm) const
        {
            return const_cast<dpd_marray_base&>(*this).permuted(perm);
        }

        dpd_marray_view<Type,NDim> permuted(const detail::array_1d<unsigned>& perm)
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

        template <typename... Irreps>
        detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value &&
                            sizeof...(Irreps) == NDim, marray_view<ctype, NDim>>
        operator()(const Irreps&... irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps...);
        }

        template <typename... Irreps>
        detail::enable_if_t<detail::are_assignable<unsigned&, Irreps...>::value &&
                            sizeof...(Irreps) == NDim, marray_view<Type, NDim>>
        operator()(const Irreps&... irreps)
        {
            return const_cast<dpd_marray_base&>(*this)({(unsigned)irreps...});
        }

        marray_view<ctype, NDim> operator()(const detail::array_1d<unsigned>& irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps);
        }

        marray_view<Type, NDim> operator()(const detail::array_1d<unsigned>& irreps_)
        {
            MARRAY_ASSERT(irreps_.size() == NDim);

            std::array<unsigned, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            irreps_.slurp(irreps);

            unsigned irrep = 0;
            for (auto& i: irreps) irrep ^= i;
            MARRAY_ASSERT(irrep == irrep_);

            pointer ptr = data();
            this->get_block(irreps, len, ptr, stride);

            return marray_view<Type, NDim>(len, ptr, stride);
        }

        /***********************************************************************
         *
         * Slicing
         *
         **********************************************************************/

        template <typename... Slices>
        detail::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value &&
                            (detail::sliced_dimension<Slices...>::value > 0) &&
                            (detail::sliced_dimension<Slices...>::value < NDim),
                            dpd_marray_view<ctype, detail::sliced_dimension<Slices...>::value>>
        operator()(const Slices&... slices) const
        {
            return const_cast<dpd_marray_base&>(*this)(slices...);
        }

        template <typename... Slices>
        detail::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value &&
                            (detail::sliced_dimension<Slices...>::value > 0) &&
                            (detail::sliced_dimension<Slices...>::value < NDim),
                            dpd_marray_view<Type, detail::sliced_dimension<Slices...>::value>>
        operator()(const Slices&... slices)
        {
            constexpr unsigned NDim2 = detail::sliced_dimension<Slices...>::value;

            abort();
            //TODO
        }

        template <typename... Slices>
        detail::enable_if_t<detail::are_assignable<dpd_range_t&, Slices...>::value &&
                            sizeof...(Slices) == NDim, dpd_marray_view<ctype, NDim>>
        operator()(const Slices&... slices) const
        {
            return const_cast<dpd_marray_base&>(*this)(slices...);
        }

        template <typename... Slices>
        detail::enable_if_t<detail::are_assignable<dpd_range_t&, Slices...>::value &&
                            sizeof...(Slices) == NDim, dpd_marray_view<Type, NDim>>
        operator()(const Slices&... slices)
        {
            return const_cast<dpd_marray_base&>(*this)({slices...});
        }

        dpd_marray_view<ctype, NDim> operator()(const detail::array_1d<dpd_range_t>& slices) const
        {
            return const_cast<dpd_marray_base&>(*this)(slices);
        }

        dpd_marray_view<Type, NDim> operator()(const detail::array_1d<dpd_range_t>& slices_)
        {
            MARRAY_ASSERT(slices_.size() == NDim);

            std::array<dpd_range_t, NDim> slices;
            slices_.slurp(slices);

            dpd_marray_view<Type, NDim> v = view();

            for (unsigned i = 0;i < NDim;i++)
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
            return length(Dim, irrep);
        }

        len_type length(unsigned dim, unsigned irrep) const
        {
            MARRAY_ASSERT(irrep < nirrep_);
            return lengths(dim)[irrep];
        }

        template <unsigned Dim>
        const std::array<len_type,8>& lengths() const
        {
            static_assert(Dim < NDim, "Dim out of range");
            return lengths(Dim);
        }

        const std::array<len_type,8>& lengths(unsigned dim) const
        {
            MARRAY_ASSERT(dim < NDim);
            return len_[perm_[dim]];
        }

        std::array<std::array<len_type,8>, NDim> lengths() const
        {
            std::array<std::array<len_type,8>, NDim> len = {};
            for (unsigned i = 0;i < NDim;i++) len[i] = lengths(i);
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
