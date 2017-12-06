#ifndef _TBLIS_DIAG_SCALED_MATRIX_HPP_
#define _TBLIS_DIAG_SCALED_MATRIX_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T, int RowCol>
class diag_scaled_matrix_view
{
    public:
        enum { COL_SCALED, ROW_SCALED };

    protected:
        T* data_ = nullptr;
        std::array<len_type,2> len_ = {};
        std::array<stride_type,2> stride_ = {};
        T* diag_ = nullptr;
        stride_type diag_stride_ = 0;

    public:
        diag_scaled_matrix_view() {}

        diag_scaled_matrix_view(len_type m, len_type n, T* ptr, stride_type rs, stride_type cs, T* diag, stride_type inc)
        : data_(ptr), len_{m, n}, stride_{rs, cs}, diag_(diag), diag_stride_(inc) {}

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);

            data_ += n*stride_[dim];
            if (dim == RowCol) diag_ += n*diag_stride_;
        }

        T* data() const
        {
            return data_;
        }

        T* diag() const
        {
            return diag_;
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return len_[dim];
        }

        len_type length(unsigned dim, len_type len)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(len, len_[dim]);
            return len;
        }

        const std::array<len_type, 2>& lengths() const
        {
            return len_;
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return stride_[dim];
        }

        stride_type stride(unsigned dim, stride_type stride)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(stride, stride_[dim]);
            return stride;
        }

        const std::array<stride_type, 2>& strides() const
        {
            return stride_;
        }

        stride_type diag_stride() const
        {
            return diag_stride_;
        }

        stride_type diag_stride(stride_type stride)
        {
            std::swap(stride, diag_stride_);
            return stride;
        }

        unsigned dimension() const
        {
            return 2;
        }
};

}

#endif
