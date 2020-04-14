#ifndef _MARRAY_DPD_MARRAY_BASE_HPP_
#define _MARRAY_DPD_MARRAY_BASE_HPP_

#include <climits>

#include "marray.hpp"
#include "dpd_range.hpp"

namespace MArray
{

template <typename Type, int NDim, typename Derived, bool Owner>
class dpd_marray_base;

template <typename Type, int NDim>
class dpd_marray_view;

template <typename Type, int NDim, typename Allocator=std::allocator<Type>>
class dpd_marray;

template <typename Type, typename Derived, bool Owner>
class dpd_varray_base;

template <typename Type>
class dpd_varray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class dpd_varray;

class irrep_iterator
{
    protected:
        const unsigned irrep_;
        const int ndim_;
        const unsigned irrep_bits_;
        const unsigned irrep_mask_;
        const unsigned nblock_;
        unsigned block_;

    public:
        irrep_iterator(int irrep, int nirrep, int ndim)
        : irrep_(irrep), ndim_(ndim), irrep_bits_(__builtin_popcount(nirrep-1)),
          irrep_mask_(nirrep-1), nblock_(1u << (irrep_bits_*(ndim-1)))
        {
            MARRAY_ASSERT(ndim > 0);
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);
            MARRAY_ASSERT(irrep >= 0 && irrep < nirrep);
            reset();
        }

        bool next()
        {
            bool done = ++block_ == nblock_;
            if (done) reset();
            return !done;
        }

        int nblock() const { return nblock_; }

        void block(int block)
        {
            MARRAY_ASSERT(block >= 0 && block < (int)nblock_);
            block_ = block;
        }

        int block() const
        {
            return block_;
        }

        void reset()
        {
            block_ = UINT_MAX;
        }

        int irrep(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < ndim_);

            if (dim == 0)
            {
                auto irr0 = irrep_;
                for (auto i : range(ndim_-1))
                    irr0 ^= (block_ >> irrep_bits_*i) & irrep_mask_;
                return irr0;
            }

            return (block_ >> irrep_bits_*(dim-1)) & irrep_mask_;
        }
};

namespace detail
{

template <typename Derived>
struct dpd_base
{
    static dim_vector default_depth(dpd_layout layout, int ndim)
    {
        dim_vector depth;

        auto log2_floor = [&](int x)
        {
            return sizeof(x)*CHAR_BIT - __builtin_clz(x) - 1;
        };

        auto is_power_of_two = [&](int x)
        {
            return (x & (x-1)) == 0;
        };

        auto log2_ceil = [&](int x)
        {
            return log2_floor(x) + (is_power_of_two(x) ? 0 : 1);
        };

        if (layout == BALANCED_ROW_MAJOR ||
            layout == BALANCED_COLUMN_MAJOR)
        {
            int dl = log2_floor(ndim);
            int du = log2_ceil(ndim);
            int o = ndim - (1<<dl);
            int k = 2*((o+1)/2);
            int l = 2*(o/2);
            int h = (ndim+1)/2;

            depth.insert(depth.end(), k, du);
            depth.insert(depth.end(), h-k, dl);
            depth.insert(depth.end(), l, du);
            depth.insert(depth.end(), ndim-h-l, dl);
        }
        else if (layout == PREFIX_ROW_MAJOR ||
                 layout == PREFIX_COLUMN_MAJOR)
        {
            depth.push_back(ndim-1);
            for (auto i : range(1,ndim)) depth.push_back(ndim-i);
        }
        else //if (layout == BLOCKED_ROW_MAJOR ||
             //    layout == BLOCKED_COLUMN_MAJOR)
        {
            for (auto i : range(1,ndim)) depth.push_back(i);
            depth.push_back(ndim-1);
        }

        return depth;
    }

    void set_tree()
    {
        auto& leaf = derived().leaf_;
        auto& parent = derived().parent_;
        int ndim = leaf.size();
        dim_vector depth(derived().depth_.begin(), derived().depth_.end());
        dim_vector node(ndim);
        len_vector leaf_idx = range(ndim);

        MARRAY_ASSERT((int)depth.size() == ndim);
        for (auto i : depth)
            MARRAY_ASSERT(i < ndim);

        auto pos = 0;
        for (auto d : reversed_range(ndim))
        {
            auto i = 0;
            while (i < (int)depth.size())
            {
                /*
                 * If we encounter a pair of nodes with depth greater than the
                 * current level, combine them into an interior node and
                 * assign the parent pointers.
                 */
                if (depth[i] == d+1)
                {
                    MARRAY_ASSERT(i+1 < (int)depth.size() && depth[i+1] == d+1);
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

                i++;
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
        int ndim = perm.size();

        if (layout == COLUMN_MAJOR)
        {
            // Column major
            for (auto i : range(ndim))
            {
                std::copy_n(&len[i][0], nirrep, &size[leaf[i]][0]);
                perm[i] = i;
            }
        }
        else
        {
            // Row major: reverse the dimensions and treat as
            // permuted column major
            for (auto i : range(ndim))
            {
                std::copy_n(&len[i][0], nirrep, &size[leaf[ndim-1-i]][0]);
                perm[i] = ndim-1-i;
            }

            for (auto i : range(ndim/2))
            for (auto j : range(nirrep))
                std::swap(len[i][j], len[ndim-1-i][j]);
        }

        for (auto i : range(ndim-1))
        {
            auto next = parent[2*i];

            for (auto irr1 : range(nirrep))
            {
                size[next][irr1] = 0;
                for (auto irr2 : range(nirrep))
                    size[next][irr1] += size[2*i][irr1^irr2]*size[2*i+1][irr2];
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
        int ndim = perm.size();

        short_vector<int, 2*MARRAY_OPT_NDIM-1> dpd_irrep(2*ndim-1);
        short_vector<stride_type, 2*MARRAY_OPT_NDIM-1> dpd_stride(2*ndim-1);
        dpd_stride[2*ndim-2] = 1;

        auto it = irreps.begin();
        for (auto i : range(ndim))
        {
            dpd_irrep[leaf[perm[i]]] = *it;
            ++it;
        }

        for (auto i : range(ndim-1))
            dpd_irrep[parent[2*i]] = dpd_irrep[2*i]^dpd_irrep[2*i+1];

        for (auto i : reversed_range(ndim-1))
        {
            auto irrep = dpd_irrep[parent[2*i]];

            dpd_stride[2*i] = dpd_stride[parent[2*i]];
            dpd_stride[2*i+1] = dpd_stride[2*i]*size[2*i][dpd_irrep[2*i]];

            stride_type offset = 0;
            for (auto irr1 : range(dpd_irrep[2*i+1]))
                offset += size[2*i][irr1^irrep]*size[2*i+1][irr1];

            data += offset*dpd_stride[2*i];
        }

        it = irreps.begin();
        auto l = len.begin();
        auto s = stride.begin();
        for (auto i : range(ndim))
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

template <typename Type, int NDim, typename Derived, bool Owner>
class dpd_marray_base : protected detail::dpd_base<dpd_marray_base<Type, NDim, Derived, Owner>>
{
    static_assert(NDim > 0, "NDim must be positive");

    template <typename> friend struct detail::dpd_base;
    template <typename, int, typename, bool> friend class dpd_marray_base;
    template <typename, int> friend class dpd_marray_view;
    template <typename, int, typename> friend class dpd_marray;

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
        std::array<int, NDim> leaf_ = {};
        std::array<int, 2*NDim-1> parent_ = {};
        std::array<int, NDim> perm_ = {};
        std::array<int, NDim> depth_ = {};
        pointer data_ = nullptr;
        int irrep_ = 0;
        int nirrep_ = 0;
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

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, pointer ptr,
                   dpd_layout layout = DEFAULT)
        {
            reset(irrep, nirrep, len, ptr,
                  this->default_depth(layout, NDim), layout.base());
        }

        void reset(int irrep, int nirrep,
                   const detail::array_2d<len_type>& len, pointer ptr,
                   const detail::array_1d<int>& depth, layout layout = DEFAULT)
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

        template <typename View, typename Func, int... I>
        void for_each_block(Func&& f, detail::integer_sequence<int, I...>) const
        {
            typedef typename View::pointer Ptr;

            const_pointer cptr;
            std::array<int, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            irrep_iterator it(irrep_, nirrep_, NDim);
            while (it.next())
            {
                irreps[0] = irrep_;
                for (auto i : range(1,NDim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                auto empty = false;
                for (auto i : range(NDim))
                    if (length(i, irreps[i]) == 0)
                        empty = true;
                if (empty) continue;

                cptr = data();
                this->get_block(irreps, len, cptr, stride);

                detail::call(std::forward<Func>(f),
                             View(len, const_cast<Ptr>(cptr), stride),
                             irreps[I]...);
            }
        }

        template <typename Tp, typename Func, int... I>
        void for_each_element(Func&& f, detail::integer_sequence<int, I...>) const
        {
            typedef Tp* Ptr;

            const_pointer cptr;
            std::array<int, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            irrep_iterator it(irrep_, nirrep_, NDim);
            while (it.next())
            {
                irreps[0] = irrep_;
                for (auto i : range(1,NDim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                bool empty = false;
                for (auto i : range(NDim))
                    if (length(i, irreps[i]) == 0)
                        empty = true;
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

        static stride_type size(int irrep, const detail::array_2d<len_type>& len_)
        {
            if (len_.length(0) == 0) return 1;

            //TODO: add alignment option

            matrix<len_type> len;
            len_.slurp(len);

            auto nirrep = len.length(1);
            auto ndim = len.length(0);

            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            std::array<stride_type, 8> size;

            for (auto irr : range(nirrep))
                size[irr] = len[0][irr];

            for (auto i : range(1,ndim))
            {
                std::array<stride_type, 8> new_size = {};

                for (auto irr1 : range(nirrep))
                for (auto irr2 : range(nirrep))
                    new_size[irr1] += size[irr1^irr2]*len[i][irr2];

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

            for (auto i : range(NDim))
                MARRAY_ASSERT(lengths(i) == other.lengths(i));

            irrep_iterator it(irrep_, nirrep_, NDim);
            std::array<int, NDim> irreps;
            while (it.next())
            {
                irreps[0] = irrep_;
                for (auto i : range(1,NDim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                (*this)(irreps) = other(irreps);
            }

            return static_cast<Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            irrep_iterator it(irrep_, nirrep_, NDim);
            std::array<int, NDim> irreps;
            while (it.next())
            {
                irreps[0] = irrep_;
                for (auto i : range(1,NDim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

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

        dpd_marray_view<ctype,NDim> permuted(const detail::array_1d<int>& perm) const
        {
            return const_cast<dpd_marray_base&>(*this).permuted(perm);
        }

        dpd_marray_view<Type,NDim> permuted(const detail::array_1d<int>& perm)
        {
            dpd_marray_view<Type,NDim> r(*this);
            r.permute(perm);
            return r;
        }

        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<ctype, NDim> transposed() const
        {
            return const_cast<dpd_marray_base&>(*this).transposed();
        }

        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<Type, NDim> transposed()
        {
            return permuted({1, 0});
        }

        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        dpd_marray_view<ctype, NDim> T() const
        {
            return const_cast<dpd_marray_base&>(*this).T();
        }

        template <int N=NDim, typename=detail::enable_if_t<N==2>>
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
        detail::enable_if_t<detail::are_assignable<int&, Irreps...>::value &&
                            sizeof...(Irreps) == NDim, marray_view<ctype, NDim>>
        operator()(const Irreps&... irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps...);
        }

        template <typename... Irreps>
        detail::enable_if_t<detail::are_assignable<int&, Irreps...>::value &&
                            sizeof...(Irreps) == NDim, marray_view<Type, NDim>>
        operator()(const Irreps&... irreps)
        {
            return const_cast<dpd_marray_base&>(*this)({(int)irreps...});
        }

        marray_view<ctype, NDim> operator()(const detail::array_1d<int>& irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps);
        }

        marray_view<Type, NDim> operator()(const detail::array_1d<int>& irreps_)
        {
            MARRAY_ASSERT(irreps_.size() == NDim);

            std::array<int, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            irreps_.slurp(irreps);

            int irrep = 0;
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
        operator()(const Slices&...)
        {
            constexpr int NDim2 = detail::sliced_dimension<Slices...>::value;

            (void)NDim2;

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

            for (auto i : range(NDim))
            for (auto j : range(nirrep_))
            {
                v.off_[i][j] = slices[i].from(j);
                v.len_[i][j] = slices[i].size(j);
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
            for_each_block<marray_view<ctype, NDim>>(std::forward<Func>(f), detail::static_range<int, NDim>{});
        }

        template <typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block<marray_view<Type, NDim>>(std::forward<Func>(f), detail::static_range<int, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<ctype>(std::forward<Func>(f), detail::static_range<int, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<Type>(std::forward<Func>(f), detail::static_range<int, NDim>{});
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

        template <int Dim>
        len_type length(int irrep) const
        {
            static_assert(Dim >= 0 && Dim < NDim, "Dim out of range");
            return length(Dim, irrep);
        }

        len_type length(int dim, int irrep) const
        {
            MARRAY_ASSERT(irrep >= 0 && irrep < nirrep_);
            return lengths(dim)[irrep];
        }

        template <int Dim>
        const std::array<len_type,8>& lengths() const
        {
            static_assert(Dim <= 0 && Dim < NDim, "Dim out of range");
            return lengths(Dim);
        }

        const std::array<len_type,8>& lengths(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            return len_[perm_[dim]];
        }

        std::array<std::array<len_type,8>, NDim> lengths() const
        {
            std::array<std::array<len_type,8>, NDim> len = {};
            for (auto i : range(NDim)) len[i] = lengths(i);
            return len;
        }

        auto irrep() const
        {
            return irrep_;
        }

        auto num_irreps() const
        {
            return nirrep_;
        }

        auto& permutation() const
        {
            return perm_;
        }

        static constexpr auto dimension()
        {
            return NDim;
        }
};

}

#endif
