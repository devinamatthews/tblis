#include <tblis/internal/indexed_dpd.hpp>

namespace tblis
{
namespace internal
{

void shift(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, const scalar& beta, bool conj_A,
           const indexed_dpd_marray_view<char>& A, const dim_vector& idx_A_A)
{
    auto local_A = A[0];

    for (len_type i = 0;i < A.num_indices();i++)
    {
        scalar alpha_fac = alpha;

        switch (type)
        {
            case FLOAT:    alpha_fac.data.s *= reinterpret_cast<const indexed_dpd_marray_view<   float>&>(A).factor(i); break;
            case DOUBLE:   alpha_fac.data.d *= reinterpret_cast<const indexed_dpd_marray_view<  double>&>(A).factor(i); break;
            case SCOMPLEX: alpha_fac.data.c *= reinterpret_cast<const indexed_dpd_marray_view<scomplex>&>(A).factor(i); break;
            case DCOMPLEX: alpha_fac.data.z *= reinterpret_cast<const indexed_dpd_marray_view<dcomplex>&>(A).factor(i); break;
        }

        local_A.data(A.data(i));

        if (alpha_fac.is_zero())
        {
            if (beta.is_zero())
            {
                set(type, comm, cfg, beta, local_A, idx_A_A);
            }
            else if (!beta.is_one() || (beta.is_complex() && conj_A))
            {
                scale(type, comm, cfg, beta, conj_A, local_A, idx_A_A);
            }
        }
        else
        {
            shift(type, comm, cfg, alpha_fac, beta, conj_A, local_A, idx_A_A);
        }
    }
}

}
}
