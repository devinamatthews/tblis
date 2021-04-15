#include <tblis/internal/indexed_dpd.hpp>

namespace tblis
{
namespace internal
{

void scale(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, bool conj_A, const indexed_dpd_varray_view<char>& A,
           const dim_vector& idx_A_A)
{
    auto local_A = A[0];

    for (len_type i = 0;i < A.num_indices();i++)
    {
        scalar alpha_fac = alpha;

        switch (type)
        {
            case FLOAT:    alpha_fac.data.s *= reinterpret_cast<const indexed_dpd_varray_view<   float>&>(A).factor(i); break;
            case DOUBLE:   alpha_fac.data.d *= reinterpret_cast<const indexed_dpd_varray_view<  double>&>(A).factor(i); break;
            case SCOMPLEX: alpha_fac.data.c *= reinterpret_cast<const indexed_dpd_varray_view<scomplex>&>(A).factor(i); break;
            case DCOMPLEX: alpha_fac.data.z *= reinterpret_cast<const indexed_dpd_varray_view<dcomplex>&>(A).factor(i); break;
        }

        local_A.data(A.data(i));

        if (alpha_fac.is_zero())
        {
            set(type, comm, cfg, alpha_fac, local_A, idx_A_A);
        }
        else
        {
            scale(type, comm, cfg, alpha_fac, conj_A, local_A, idx_A_A);
        }
    }
}

}
}
