#include <tblis/internal/indexed.hpp>
#include <tblis/internal/dpd.hpp>

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, const indexed_marray_view<char>& A, const dim_vector&)
{
    for (len_type i = 0;i < A.num_indices();i++)
    {
        scalar alpha_fac = alpha;

        switch (type)
        {
            case FLOAT:    alpha_fac.data.s *= reinterpret_cast<const indexed_marray_view<   float>&>(A).factor(i); break;
            case DOUBLE:   alpha_fac.data.d *= reinterpret_cast<const indexed_marray_view<  double>&>(A).factor(i); break;
            case SCOMPLEX: alpha_fac.data.c *= reinterpret_cast<const indexed_marray_view<scomplex>&>(A).factor(i); break;
            case DCOMPLEX: alpha_fac.data.z *= reinterpret_cast<const indexed_marray_view<dcomplex>&>(A).factor(i); break;
        }

        set(type, comm, cfg, A.dense_lengths(), alpha_fac, A.data(i),
            A.dense_strides());
    }
}

}
}
