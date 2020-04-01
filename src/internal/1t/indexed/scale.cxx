#include "scale.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void scale(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, bool conj_A, const indexed_varray_view<char>& A,
           const dim_vector&)
{
    for (len_type i = 0;i < A.num_indices();i++)
    {
        scalar alpha_fac = alpha;

        switch (type)
        {
            case TYPE_FLOAT:    alpha_fac.data.s *= reinterpret_cast<const indexed_varray_view<   float>&>(A).factor(i); break;
            case TYPE_DOUBLE:   alpha_fac.data.d *= reinterpret_cast<const indexed_varray_view<  double>&>(A).factor(i); break;
            case TYPE_SCOMPLEX: alpha_fac.data.c *= reinterpret_cast<const indexed_varray_view<scomplex>&>(A).factor(i); break;
            case TYPE_DCOMPLEX: alpha_fac.data.z *= reinterpret_cast<const indexed_varray_view<dcomplex>&>(A).factor(i); break;
        }

        if (alpha_fac.is_zero())
        {
            set(type, comm, cfg, A.dense_lengths(), alpha_fac, A.data(i),
                A.dense_strides());
        }
        else
        {
            scale(type, comm, cfg, A.dense_lengths(), alpha_fac, conj_A, A.data(i),
                  A.dense_strides());
        }
    }
}

}
}
