#include <tblis/internal/dpd.hpp>

namespace tblis
{
namespace internal
{

template <typename T>
void add_full(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, const dpd_marray_view<T>& A,
               const dim_vector& idx_A_A,
               const dim_vector& idx_A_AB,
               T  beta, bool conj_B, const dpd_marray_view<T>& B,
               const dim_vector& idx_B_B,
               const dim_vector& idx_B_AB)
{
    marray<T> A2, B2;

    comm.broadcast(
    [&](marray<T>& A2, marray<T>& B2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);

        auto len_A = stl_ext::select_from(A2.lengths(), idx_A_A);
        auto len_B = stl_ext::select_from(B2.lengths(), idx_B_B);
        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto stride_A_A = stl_ext::select_from(A2.strides(), idx_A_A);
        auto stride_B_B = stl_ext::select_from(B2.strides(), idx_B_B);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);

        add(type_tag<T>::value, comm, cfg, len_A, len_B, len_AB,
            alpha, conj_A, reinterpret_cast<char*>(A2.data()), stride_A_A, stride_A_AB,
             beta, conj_B, reinterpret_cast<char*>(B2.data()), stride_B_B, stride_B_AB);

        full_to_block(comm, cfg, B2, B);
    },
    A2, B2);
}

static
void trace_block(type_t type, const communicator& comm, const config& cfg,
                 const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                 const dim_vector& idx_A,
                 const dim_vector& idx_A_AB,
                 const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
                 const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep_AB = B.irrep();
    const auto irrep_A = A.irrep()^irrep_AB;
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const int ndim_A_only = idx_A.size();
    const int ndim_AB = idx_A_AB.size();

    stride_type nblock_A = ipow(nirrep, ndim_A_only-1);
    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                      irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(B, irreps_B)) continue;

        marray_view<char> local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_B.lengths(), idx_B_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        auto local_beta = beta;
        auto local_conj_B = conj_B;

        for (stride_type block_A = 0;block_A < nblock_A;block_A++)
        {
            assign_irreps(ndim_A_only, irrep_A, nirrep, block_A,
                  irreps_A, idx_A);

            if (is_block_empty(A, irreps_A)) continue;

            marray_view<char> local_A = A(irreps_A);

            auto len_A_only = stl_ext::select_from(local_A.lengths(), idx_A);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_A_A = stl_ext::select_from(local_A.strides(), idx_A);

            add(type, comm, cfg, len_A_only, {}, len_AB,
                     alpha,       conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_A, stride_A_AB,
                local_beta, local_conj_B, B.data() + (local_B.data()-B.data())*ts,         {}, stride_B_AB);

            local_beta = 1.0;
            local_conj_B = false;
        }

        if (local_beta.is_zero())
        {
            set(type, comm, cfg, local_B.lengths(),
                local_beta, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
        }
        else if (!local_beta.is_one() || (local_beta.is_complex() && local_conj_B))
        {
            scale(type, comm, cfg, local_B.lengths(),
                  local_beta, local_conj_B, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
        }
    }
}

static
void replicate_block(type_t type, const communicator& comm, const config& cfg,
                     const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                     const dim_vector& idx_A_AB,
                     const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
                     const dim_vector& idx_B,
                     const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep_AB = A.irrep();
    const auto irrep_B = B.irrep()^irrep_AB;
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const int ndim_B_only = idx_B.size();
    const int ndim_AB = idx_A_AB.size();

    stride_type nblock_B = ipow(nirrep, ndim_B_only-1);
    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        marray_view<char> local_A = A(irreps_A);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);

        for (stride_type block_B = 0;block_B < nblock_B;block_B++)
        {
            assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                          irreps_B, idx_B);

            if (is_block_empty(B, irreps_B)) continue;

            marray_view<char> local_B = B(irreps_B);

            auto len_B_only = stl_ext::select_from(local_B.lengths(), idx_B);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
            auto stride_B_B = stl_ext::select_from(local_B.strides(), idx_B);

            add(type, comm, cfg, {}, len_B_only, len_AB,
                alpha, conj_A, A.data() + (local_A.data()-A.data())*ts,         {}, stride_A_AB,
                 beta, conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_B, stride_B_AB);
        }
    }

    if (beta.is_one() && !(beta.is_complex() && conj_B)) return;

    for (auto irrep_AB : range(nirrep))
    {
        if (irrep_AB == A.irrep()) continue;

        const auto irrep_B = B.irrep()^irrep_AB;

        for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
        {
            assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                          irreps_B, idx_B_AB);

            for (stride_type block_B = 0;block_B < nblock_B;block_B++)
            {
                assign_irreps(ndim_B_only, irrep_B, nirrep, block_B,
                              irreps_B, idx_B);

                if (is_block_empty(B, irreps_B)) continue;

                marray_view<char> local_B = B(irreps_B);

                if (beta.is_zero())
                {
                    set(type, comm, cfg, local_B.lengths(),
                        beta, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
                }
                else
                {
                    scale(type, comm, cfg, local_B.lengths(),
                          beta, conj_B, B.data() + (local_B.data()-B.data())*ts, local_B.strides());
                }
            }
        }
    }
}

static
void transpose_block(type_t type, const communicator& comm, const config& cfg,
                     const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
                     const dim_vector& idx_A_AB,
                     const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
                     const dim_vector& idx_B_AB)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();
    const auto irrep_AB = A.irrep();
    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const int ndim_AB = idx_A_AB.size();

    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);

    for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
    {
        assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                      irreps_A, idx_A_AB, irreps_B, idx_B_AB);

        if (is_block_empty(A, irreps_A)) continue;

        marray_view<char> local_A = A(irreps_A);
        marray_view<char> local_B = B(irreps_B);

        auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
        auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
        auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

        add(type, comm, cfg, {}, {}, len_AB,
            alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AB,
             beta, conj_B, B.data() + (local_B.data()-B.data())*ts, {}, stride_B_AB);
    }
}

void add(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB)
{
    if (dpd_impl == FULL)
    {
        switch (type)
        {
            case FLOAT:
                add_full(comm, cfg,
                         alpha.data.s, conj_A, reinterpret_cast<const dpd_marray_view<float>&>(A), idx_A, idx_A_AB,
                          beta.data.s, conj_B, reinterpret_cast<const dpd_marray_view<float>&>(B), idx_B, idx_B_AB);
                break;
            case DOUBLE:
                add_full(comm, cfg,
                         alpha.data.d, conj_A, reinterpret_cast<const dpd_marray_view<double>&>(A), idx_A, idx_A_AB,
                          beta.data.d, conj_B, reinterpret_cast<const dpd_marray_view<double>&>(B), idx_B, idx_B_AB);
                break;
            case SCOMPLEX:
                add_full(comm, cfg,
                         alpha.data.c, conj_A, reinterpret_cast<const dpd_marray_view<scomplex>&>(A), idx_A, idx_A_AB,
                          beta.data.c, conj_B, reinterpret_cast<const dpd_marray_view<scomplex>&>(B), idx_B, idx_B_AB);
                break;
            case DCOMPLEX:
                add_full(comm, cfg,
                         alpha.data.z, conj_A, reinterpret_cast<const dpd_marray_view<dcomplex>&>(A), idx_A, idx_A_AB,
                          beta.data.z, conj_B, reinterpret_cast<const dpd_marray_view<dcomplex>&>(B), idx_B, idx_B_AB);
                break;
        }
    }
    else if (!idx_A.empty())
    {
        trace_block(type, comm, cfg,
                    alpha, conj_A, A, idx_A, idx_A_AB,
                     beta, conj_B, B, idx_B_AB);
    }
    else if (!idx_B.empty())
    {
        replicate_block(type, comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B, idx_B_AB);
    }
    else
    {
        transpose_block(type, comm, cfg,
                        alpha, conj_A, A, idx_A_AB,
                         beta, conj_B, B, idx_B_AB);
    }
}

}
}
