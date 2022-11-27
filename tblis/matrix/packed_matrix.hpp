#ifndef TBLIS_PACKED_MATRIX_HPP
#define TBLIS_PACKED_MATRIX_HPP

#include "abstract_matrix.hpp"

namespace tblis
{

class packed_matrix : public abstract_matrix
{
    public:
        packed_matrix(type_t type, len_type m, len_type n, void* ptr, stride_type ps)
        : abstract_matrix({1, type}, false, false, m, n, {})
        {
            set_buffer(static_cast<char*>(ptr), ps);
        }

        stride_type panel_stride() const
        {
            return get_buffer_size();
        }

        char* data() const
        {
            return get_buffer();
        }
};

}

#endif
