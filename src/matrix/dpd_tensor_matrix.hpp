#ifndef _TBLIS_DPD_TENSOR_MATRIX_HPP_
#define _TBLIS_DPD_TENSOR_MATRIX_HPP_

#include "abstract_matrix.hpp"

namespace tblis
{

class irrep_iterator
{
    protected:
        const unsigned irrep_;
        const unsigned irrep_bits_;
        const unsigned irrep_mask_;
        viterator<0> it_;

    public:
        irrep_iterator(unsigned irrep, unsigned nirrep, unsigned ndim)
        : irrep_(irrep), irrep_bits_(__builtin_popcount(nirrep-1)),
          irrep_mask_ (nirrep-1), it_(irrep_vector(ndim ? ndim-1 : 0, nirrep)) {}

        bool next()
        {
            return it_.next();
        }

        unsigned nblock() const
        {
            return 1u << (irrep_bits_*it_.dimension());
        }

        void block(unsigned b)
        {
            irrep_vector irreps(it_.dimension());

            for (unsigned i = 0;i < it_.dimension();i++)
            {
                irreps[i] = b & irrep_mask_;
                b >>= irrep_bits_;
            }

            it_.position(irreps);
        }

        void reset()
        {
            it_.reset();
        }

        unsigned irrep(unsigned dim)
        {
            TBLIS_ASSERT(dim <= it_.dimension());

            if (dim == 0)
            {
                unsigned irr0 = irrep_;
                for (unsigned irr : it_.position()) irr0 ^= irr;
                return irr0;
            }

            return it_.position()[dim-1];
        }
};

template <typename T>
class dpd_tensor_matrix : public abstract_matrix<T>
{
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::off_;
        dpd_varray_view<T>& tensor_;
        std::array<dim_vector, 2> dims_ = {};
        std::array<unsigned, 2> irrep_ = {};
        std::array<unsigned, 2> block_ = {};
        std::array<len_vector, 2> block_size_ = {};
        std::array<len_vector, 2> block_idx_ = {};
        std::array<len_type, 2> block_offset_ = {};
        std::array<stride_type, 2> leading_stride_ = {};

    public:
        dpd_tensor_matrix();

        template <typename U, typename V>
        dpd_tensor_matrix(dpd_varray_view<T>& other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned col_irrep)
        : tensor_(other)
        {
            TBLIS_ASSERT(row_inds.size()+col_inds.size() == other.dimension());

            const unsigned nirrep = other.num_irreps();

            data_ = other.data();
            dims_[0].assign(row_inds.begin(), row_inds.end());
            dims_[1].assign(col_inds.begin(), col_inds.end());
            irrep_ = {col_irrep^other.irrep(), col_irrep};

            for (unsigned dim : {0,1})
            {
                if (dims_[dim].empty())
                {
                    tot_len_[dim] = irrep_[dim] == 0 ? 1 : 0;
                    block_size_[dim].push_back(tot_len_[dim]);
                    block_idx_[dim].push_back(0);
                }
                else
                {
                    tot_len_[dim] = 0;
                    irrep_iterator it(irrep_[dim], nirrep, dims_[dim].size());
                    for (unsigned idx = 0;it.next();idx++)
                    {
                        stride_type size = 1;
                        for (unsigned i = 0;i < dims_[dim].size();i++)
                            size *= other.length(dims_[dim][i], it.irrep(i));

                        if (size == 0) continue;

                        block_size_[dim].push_back(size);
                        block_idx_[dim].push_back(idx);
                        tot_len_[dim] += size;
                    }
                }
            }

            cur_len_ = tot_len_;

            len_vector stride(other.dimension(), 1);

            for (unsigned i = 0;i < other.dimension();i++)
            {
                for (unsigned irrep = 0;irrep < nirrep;irrep++)
                    stride[other.permutation()[i]] += other.length(i, irrep);
                if (i > 0) stride[i] *= stride[i-1];
            }

            leading_stride_ = {row_inds.empty() ? 1 : stride[other.permutation()[row_inds[0]]],
                               col_inds.empty() ? 1 : stride[other.permutation()[col_inds[0]]]};
        }

        template <typename U, typename V>
        dpd_tensor_matrix(dpd_varray_view<const T>& other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned col_irrep)
        : dpd_tensor_matrix(reinterpret_cast<dpd_varray_view<T>&>(other),
                            row_inds, col_inds, col_irrep) {}

        dpd_tensor_matrix& operator=(const dpd_tensor_matrix& other) = delete;

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(dims_[0], dims_[1]);
            swap(irrep_[0], irrep_[1]);
            swap(block_[0], block_[1]);
            swap(block_size_[0], block_size_[1]);
            swap(block_idx_[0], block_idx_[1]);
            swap(block_offset_[0], block_offset_[1]);
            swap(leading_stride_[0], leading_stride_[1]);
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return leading_stride_[dim];
        }

        void shift(unsigned dim, len_type n)
        {
            abstract_matrix<T>::shift(dim, n);

            n += block_offset_[dim];
            block_offset_[dim] = 0;

            while (n < 0)
                n += block_size_[dim][--block_[dim]];

            while (n > 0 && n >= block_size_[dim][block_[dim]])
                n -= block_size_[dim][block_[dim]++];

            TBLIS_ASSERT(n >= 0);
            block_offset_[dim] = n;
        }

        unsigned num_patches(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim].size();
        }
};

}

#endif
