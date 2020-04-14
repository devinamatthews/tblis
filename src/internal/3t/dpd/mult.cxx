#include "mult.hpp"

#include "internal/0/add.hpp"
#include "internal/0/mult.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1t/dpd/util.hpp"
#include "internal/1t/dpd/add.hpp"
#include "internal/1t/dpd/dot.hpp"
#include "internal/1t/dpd/scale.hpp"
#include "internal/1t/dpd/set.hpp"
#include "internal/3t/dense/mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "matrix/dpd_tensor_matrix.hpp"

#include "nodes/gemm.hpp"

#include <atomic>

namespace tblis
{
namespace internal
{

std::atomic<long> flops;
dpd_impl_t dpd_impl = BLIS;

template <typename T>
void mult_full(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, const dpd_varray_view<T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        bool conj_B, const dpd_varray_view<T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
               T  beta, bool conj_C, const dpd_varray_view<T>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    varray<T> A2, B2, C2;

    comm.broadcast(
    [&](varray<T>& A2, varray<T>& B2, varray<T>& C2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);
        block_to_full(comm, cfg, C, C2);

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto len_AC = stl_ext::select_from(C2.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(C2.lengths(), idx_C_BC);
        auto len_ABC = stl_ext::select_from(C2.lengths(), idx_C_ABC);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_A_AC = stl_ext::select_from(A2.strides(), idx_A_AC);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);
        auto stride_B_BC = stl_ext::select_from(B2.strides(), idx_B_BC);
        auto stride_C_AC = stl_ext::select_from(C2.strides(), idx_C_AC);
        auto stride_C_BC = stl_ext::select_from(C2.strides(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(A2.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(B2.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(C2.strides(), idx_C_ABC);

        mult(type_tag<T>::value, comm, cfg, len_AB, len_AC, len_BC, len_ABC,
             alpha, conj_A, reinterpret_cast<char*>(A2.data()), stride_A_AB, stride_A_AC, stride_A_ABC,
                    conj_B, reinterpret_cast<char*>(B2.data()), stride_B_AB, stride_B_BC, stride_B_ABC,
              beta, conj_C, reinterpret_cast<char*>(C2.data()), stride_C_AC, stride_C_BC, stride_C_ABC);

        full_to_block(comm, cfg, C2, C);
    },
    A2, B2, C2);
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
                                          bool conj_B, const dpd_varray_view<char>& B,
               dim_vector idx_B_AB,
               const scalar& beta,  bool conj_C, const dpd_varray_view<char>& C,
               dim_vector idx_C_AC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    const auto irrep_AC = C.irrep();
    const auto irrep_AB = A.irrep()^irrep_AC;

    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_A_AC.size());
    irrep_iterator irrep_it_AB(irrep_AB, nirrep, idx_A_AB.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_AC.next())
    {
        for (auto i : range(idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        varray_view<char> local_C = C(irreps_C);

        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);

        auto local_beta = beta;
        bool local_conj_C = conj_C;

        if (irrep_AB == B.irrep())
        {
            while (irrep_it_AB.next())
            {
                for (auto i : range(idx_A_AB.size()))
                {
                    irreps_A[idx_A_AB[i]] =
                    irreps_B[idx_B_AB[i]] = irrep_it_AB.irrep(i);
                }

                varray_view<char> local_A = A(irreps_A);
                varray_view<char> local_B = B(irreps_B);

                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

                mult(type, comm, cfg, len_AB, len_AC, {}, {},
                          alpha,       conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, stride_A_AC, {},
                                       conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, {}, {},
                     local_beta, local_conj_C, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, {}, {});

                local_beta = 1;
                local_conj_C = false;
            }
        }

        if (local_beta.is_zero())
        {
            set(type, comm, cfg, local_C.lengths(), local_beta, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
        }
        else if (!local_beta.is_one() || (local_beta.is_complex() && local_conj_C))
        {
            scale(type, comm, cfg, local_C.lengths(), local_beta, local_conj_C, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
        }
    }
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
               dim_vector idx_A_AC,
                                          bool conj_B, const dpd_varray_view<char>& B,
               dim_vector idx_B_BC,
               const scalar& beta,  bool conj_C, const dpd_varray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    for (auto irrep_AC : range(nirrep))
    {
        const auto irrep_BC = C.irrep()^irrep_AC;

        irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_C_AC.size());
        irrep_iterator irrep_it_BC(irrep_BC, nirrep, idx_C_BC.size());

        irrep_vector irreps_A(A.dimension());
        irrep_vector irreps_B(B.dimension());
        irrep_vector irreps_C(C.dimension());

        while (irrep_it_AC.next())
        while (irrep_it_BC.next())
        {
            for (auto i : range(idx_A_AC.size()))
            {
                irreps_A[idx_A_AC[i]] =
                irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
            }

            for (auto i : range(idx_B_BC.size()))
            {
                irreps_B[idx_B_BC[i]] =
                irreps_C[idx_C_BC[i]] = irrep_it_BC.irrep(i);
            }

            varray_view<char> local_C = C(irreps_C);

            auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
            auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
            auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
            auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

            if (irrep_AC == A.irrep() && irrep_BC == B.irrep())
            {
                varray_view<char> local_A = A(irreps_A);
                varray_view<char> local_B = B(irreps_B);

                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                mult(type, comm, cfg, {}, len_AC, len_BC, {},
                     alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AC, {},
                            conj_B, B.data() + (local_B.data()-B.data())*ts, {}, stride_B_BC, {},
                      beta, conj_C, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, stride_C_BC, {});
            }
            else if (beta.is_zero())
            {
                set(type, comm, cfg, local_C.lengths(), beta, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
            }
            else if (!beta.is_one() || (beta.is_complex() && conj_C))
            {
                scale(type, comm, cfg, local_C.lengths(), beta, conj_C, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
            }
        }
    }
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
                                          bool conj_B, const dpd_varray_view<char>& B,
               dim_vector idx_B_AB,
               dim_vector idx_B_BC,
               const scalar& beta,  bool conj_C, const dpd_varray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC)
{
    if ((A.irrep()^B.irrep()) != C.irrep())
    {
        if (beta.is_zero())
        {
            set(type, comm, cfg, beta, C, {});
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            scale(type, comm, cfg, beta, conj_C, C, {});
        }

        return;
    }

    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB, C, idx_C_AC);

    auto perm_AC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = detail::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);

    auto unit_A_AC = unit_dim(stride[0], idx_A_AC);
    auto unit_C_AC = unit_dim(stride[2], idx_C_AC);
    auto unit_B_BC = unit_dim(stride[1], idx_B_BC);
    auto unit_C_BC = unit_dim(stride[2], idx_C_BC);
    auto unit_A_AB = unit_dim(stride[0], idx_A_AB);
    auto unit_B_AB = unit_dim(stride[1], idx_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == (int)perm_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == (int)perm_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == (int)perm_AB.size() &&
                  unit_B_AB == (int)perm_AB.size()));

    bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < (int)perm_AC.size();
    bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < (int)perm_BC.size();
    bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < (int)perm_AB.size()) ||
                     (unit_B_AB > 0 && unit_B_AB < (int)perm_AB.size());

    if (pack_M_3d)
    {
        std::rotate(idx_A_AC.begin()+1, idx_A_AC.begin()+unit_A_AC, idx_A_AC.end());
        std::rotate(idx_C_AC.begin()+1, idx_C_AC.begin()+unit_A_AC, idx_C_AC.end());
    }

    if (pack_N_3d)
    {
        std::rotate(idx_B_BC.begin()+1, idx_B_BC.begin()+unit_B_BC, idx_B_BC.end());
        std::rotate(idx_C_BC.begin()+1, idx_C_BC.begin()+unit_B_BC, idx_C_BC.end());
    }

    if (pack_K_3d)
    {
        auto unit_AB = std::max(unit_A_AB, unit_B_AB);
        std::rotate(idx_A_AB.begin()+1, idx_A_AB.begin()+unit_AB, idx_A_AB.end());
        std::rotate(idx_B_AB.begin()+1, idx_B_AB.begin()+unit_AB, idx_B_AB.end());
    }

    scalar one(1.0, type);

    for (auto irrep_AB : range(nirrep))
    {
        const auto irrep_AC = A.irrep()^irrep_AB;
        const auto irrep_BC = B.irrep()^irrep_AB;

        dpd_tensor_matrix at(alpha, conj_A, A, idx_A_AC, idx_A_AB, irrep_AB, {}, {}, {}, pack_M_3d, pack_K_3d);
        dpd_tensor_matrix bt(  one, conj_B, B, idx_B_AB, idx_B_BC, irrep_BC, {}, {}, {}, pack_K_3d, pack_N_3d);
        dpd_tensor_matrix ct( beta, conj_C, C, idx_C_AC, idx_C_BC, irrep_BC, {}, {}, {}, pack_M_3d, pack_N_3d);

        if (ct.length(0) == 0 || ct.length(1) == 0) continue;

        if (at.length(1) != 0)
        {
            GotoGEMM{}(comm, cfg, at, bt, ct);
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            irrep_iterator row_it(irrep_AC, nirrep, idx_C_AC.size());
            irrep_iterator col_it(irrep_BC, nirrep, idx_C_BC.size());

            irrep_vector irreps(C.dimension());

            while (row_it.next())
            {
                for (auto i : range(idx_C_AC.size()))
                    irreps[idx_C_AC[i]] = row_it.irrep(i);

                while (col_it.next())
                {
                    for (auto i : range(idx_C_BC.size()))
                        irreps[idx_C_BC[i]] = col_it.irrep(i);

                    varray_view<char> local_C = C(irreps);

                    if (beta.is_zero())
                    {
                        set(type, comm, cfg, local_C.lengths(),
                            beta, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
                    }
                    else
                    {
                        scale(type, comm, cfg, local_C.lengths(),
                              beta, conj_C, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
                    }
                }
            }
        }
    }
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                                          bool conj_B, const dpd_varray_view<char>& B,
               dim_vector idx_B_AB,
               dim_vector idx_B_ABC,
               const scalar&  beta, bool conj_C, const dpd_varray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    const auto irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    const auto irrep_AB = A.irrep()^C.irrep();
    const auto irrep_AC = A.irrep()^B.irrep();

    irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_A_AC.size());
    irrep_iterator irrep_it_AB(irrep_AB, nirrep, idx_A_AB.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_ABC.next())
    while (irrep_it_AC.next())
    {
        for (auto i : range(idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = irrep_it_ABC.irrep(i);
        }

        for (auto i : range(idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        varray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);

        auto local_beta = beta;
        bool local_conj_C = conj_C;

        while (irrep_it_AB.next())
        {
            for (auto i : range(idx_A_AB.size()))
            {
                irreps_A[idx_A_AB[i]] =
                irreps_B[idx_B_AB[i]] = irrep_it_AB.irrep(i);
            }

            varray_view<char> local_A = A(irreps_A);
            varray_view<char> local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
            auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
            auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            mult(type, comm, cfg, len_AB, len_AC, {}, len_ABC,
                      alpha,       conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, stride_A_AC, stride_A_ABC,
                                   conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, {}, stride_B_ABC,
                 local_beta, local_conj_C, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, {}, stride_C_ABC);

            local_beta = 1;
            local_conj_C = false;
        }
    }
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                                          bool conj_B, const dpd_varray_view<char>& B,
               dim_vector idx_B_BC,
               dim_vector idx_B_ABC,
               const scalar&  beta, bool conj_C, const dpd_varray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC,
               dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_ABC(idx_A_ABC.size());
    len_vector len_ABC(idx_A_ABC.size());

    const auto irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    const auto irrep_AC = A.irrep()^irrep_ABC;
    const auto irrep_BC = B.irrep()^irrep_ABC;

    irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_C_ABC.size());
    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_C_AC.size());
    irrep_iterator irrep_it_BC(irrep_BC, nirrep, idx_C_BC.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_ABC.next())
    while (irrep_it_AC.next())
    while (irrep_it_BC.next())
    {
        for (auto i : range(idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = irrep_it_ABC.irrep(i);
        }

        for (auto i : range(idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        for (auto i : range(idx_B_BC.size()))
        {
            irreps_B[idx_B_BC[i]] =
            irreps_C[idx_C_BC[i]] = irrep_it_BC.irrep(i);
        }

        varray_view<char> local_A = A(irreps_A);
        varray_view<char> local_B = B(irreps_B);
        varray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
        auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);
        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

        mult(type, comm, cfg, {}, len_AC, len_BC, len_ABC,
             alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AC, stride_A_ABC,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, {}, stride_B_BC, stride_B_ABC,
              beta, conj_C, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, stride_C_BC, stride_C_ABC);
    }
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                                          bool conj_B, const dpd_varray_view<char>& B,
               dim_vector idx_B_AB,
               dim_vector idx_B_BC,
               dim_vector idx_B_ABC,
               const scalar& beta,  bool conj_C, const dpd_varray_view<char>& C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC,
               dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB, C, idx_C_AC);

    auto perm_AC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = detail::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));
    auto perm_ABC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_A_ABC),
                                           stl_ext::select_from(stride[0], idx_B_ABC),
                                           stl_ext::select_from(stride[1], idx_C_ABC));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);
    stl_ext::permute(idx_A_ABC, perm_ABC);
    stl_ext::permute(idx_B_ABC, perm_ABC);
    stl_ext::permute(idx_C_ABC, perm_ABC);

    auto unit_A_AC = unit_dim(stride[0], idx_A_AC);
    auto unit_C_AC = unit_dim(stride[2], idx_C_AC);
    auto unit_B_BC = unit_dim(stride[1], idx_B_BC);
    auto unit_C_BC = unit_dim(stride[2], idx_C_BC);
    auto unit_A_AB = unit_dim(stride[0], idx_A_AB);
    auto unit_B_AB = unit_dim(stride[1], idx_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == (int)perm_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == (int)perm_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == (int)perm_AB.size() &&
                  unit_B_AB == (int)perm_AB.size()));

    bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < (int)perm_AC.size();
    bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < (int)perm_BC.size();
    bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < (int)perm_AB.size()) ||
                     (unit_B_AB > 0 && unit_B_AB < (int)perm_AB.size());

    if (pack_M_3d)
    {
        std::rotate(idx_A_AC.begin()+1, idx_A_AC.begin()+unit_A_AC, idx_A_AC.end());
        std::rotate(idx_C_AC.begin()+1, idx_C_AC.begin()+unit_A_AC, idx_C_AC.end());
    }

    if (pack_N_3d)
    {
        std::rotate(idx_B_BC.begin()+1, idx_B_BC.begin()+unit_B_BC, idx_B_BC.end());
        std::rotate(idx_C_BC.begin()+1, idx_C_BC.begin()+unit_B_BC, idx_C_BC.end());
    }

    if (pack_K_3d)
    {
        auto unit_AB = std::max(unit_A_AB, unit_B_AB);
        std::rotate(idx_A_AB.begin()+1, idx_A_AB.begin()+unit_AB, idx_A_AB.end());
        std::rotate(idx_B_AB.begin()+1, idx_B_AB.begin()+unit_AB, idx_B_AB.end());
    }

    scalar one(1.0, type);

    irrep_vector irreps_ABC(idx_A_ABC.size());
    len_vector len_ABC(idx_A_ABC.size());

    for (auto irrep_ABC : range(nirrep))
    for (auto irrep_AB : range(nirrep))
    {
        auto irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
        auto irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

        irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());

        while (irrep_it_ABC.next())
        {
            for (auto i : range(idx_A_ABC.size()))
            {
                irreps_ABC[i] = irrep_it_ABC.irrep(i);
                len_ABC[i] = A.length(idx_A_ABC[i], irreps_ABC[i]);
            }

            viterator<0> it_ABC(len_ABC);

            while (it_ABC.next())
            {
                dpd_tensor_matrix at(alpha, conj_A, A, idx_A_AC, idx_A_AB, irrep_AB, idx_A_ABC,
                                        irreps_ABC, it_ABC.position(), pack_M_3d, pack_K_3d);
                dpd_tensor_matrix bt(  one, conj_B, B, idx_B_AB, idx_B_BC, irrep_BC, idx_B_ABC,
                                        irreps_ABC, it_ABC.position(), pack_K_3d, pack_N_3d);
                dpd_tensor_matrix ct( beta, conj_B, C, idx_C_AC, idx_C_BC, irrep_BC, idx_C_ABC,
                                        irreps_ABC, it_ABC.position(), pack_M_3d, pack_N_3d);

                if (ct.length(0) == 0 || ct.length(1) == 0) continue;

                if (at.length(1) != 0)
                {
                    GotoGEMM{}(comm, cfg, at, bt, ct);
                }
                else if (!beta.is_one() || (beta.is_complex() && conj_C))
                {
                    irrep_iterator row_it(irrep_AC, nirrep, idx_C_AC.size());
                    irrep_iterator col_it(irrep_BC, nirrep, idx_C_BC.size());

                    irrep_vector irreps(C.dimension());

                    while (row_it.next())
                    {
                        for (auto i : range(idx_C_AC.size()))
                            irreps[idx_C_AC[i]] = row_it.irrep(i);

                        while (col_it.next())
                        {
                            for (auto i : range(idx_C_BC.size()))
                                irreps[idx_C_BC[i]] = col_it.irrep(i);

                            varray_view<char> local_C = C(irreps);

                            if (beta.is_zero())
                            {
                                set(type, comm, cfg, local_C.lengths(),
                                    beta, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
                            }
                            else
                            {
                                scale(type, comm, cfg, local_C.lengths(),
                                      beta, conj_C, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
                            }
                        }
                    }
                }
            }
        }
    }
}

void mult_block(type_t type, const communicator& comm, const config& cfg,
                const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                               bool conj_B, const dpd_varray_view<char>& B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                const scalar&  beta, bool conj_C, const dpd_varray_view<char>& C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    const auto ndim_A = A.dimension();
    const auto ndim_B = B.dimension();
    const auto ndim_C = C.dimension();

    const int ndim_AC = idx_C_AC.size();
    const int ndim_BC = idx_C_BC.size();
    const int ndim_AB = idx_A_AB.size();
    const int ndim_ABC = idx_A_ABC.size();

    std::array<len_vector,3> len;
    std::array<stride_vector,3> stride;
    dense_total_lengths_and_strides(len, stride, A, idx_A_AB, B, idx_B_AB,
                                    C, idx_C_AC);

    auto perm_AC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_AC),
                                          stl_ext::select_from(stride[0], idx_A_AC));
    auto perm_BC = detail::sort_by_stride(stl_ext::select_from(stride[2], idx_C_BC),
                                          stl_ext::select_from(stride[1], idx_B_BC));
    auto perm_AB = detail::sort_by_stride(stl_ext::select_from(stride[0], idx_A_AB),
                                          stl_ext::select_from(stride[1], idx_B_AB));

    stl_ext::permute(idx_A_AC, perm_AC);
    stl_ext::permute(idx_A_AB, perm_AB);
    stl_ext::permute(idx_B_AB, perm_AB);
    stl_ext::permute(idx_B_BC, perm_BC);
    stl_ext::permute(idx_C_AC, perm_AC);
    stl_ext::permute(idx_C_BC, perm_BC);

    stride_type nblock_AB = ipow(nirrep, ndim_AB-1);
    stride_type nblock_AC = ipow(nirrep, ndim_AC-1);
    stride_type nblock_BC = ipow(nirrep, ndim_BC-1);
    stride_type nblock_ABC = ipow(nirrep, ndim_ABC-1);

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);
    irrep_vector irreps_C(ndim_C);

    for (auto irrep_ABC : range(nirrep))
    {
        if (ndim_ABC == 0 && irrep_ABC != 0) continue;

        for (auto irrep_AB : range(nirrep))
        {
            auto irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
            auto irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

            if (ndim_AC == 0 && irrep_AC != 0) continue;
            if (ndim_BC == 0 && irrep_BC != 0) continue;

            for (stride_type block_ABC = 0;block_ABC < nblock_ABC;block_ABC++)
            {
                assign_irreps(ndim_ABC, irrep_ABC, nirrep, block_ABC,
                              irreps_A, idx_A_ABC, irreps_B, idx_B_ABC, irreps_C, idx_C_ABC);

                for (stride_type block_AC = 0;block_AC < nblock_AC;block_AC++)
                {
                    assign_irreps(ndim_AC, irrep_AC, nirrep, block_AC,
                                  irreps_A, idx_A_AC, irreps_C, idx_C_AC);

                    for (stride_type block_BC = 0;block_BC < nblock_BC;block_BC++)
                    {
                        assign_irreps(ndim_BC, irrep_BC, nirrep, block_BC,
                                      irreps_B, idx_B_BC, irreps_C, idx_C_BC);

                        if (is_block_empty(C, irreps_C)) continue;

                        varray_view<char> local_C = C(irreps_C);

                        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
                        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
                        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
                        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
                        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
                        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

                        auto local_beta = beta;
                        bool local_conj_C = conj_C;

                        if ((ndim_AB != 0 || irrep_AB == 0) &&
                            irrep_ABC == (A.irrep()^B.irrep()^C.irrep()))
                        {
                            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                            {
                                assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                                if (is_block_empty(A, irreps_A)) continue;

                                varray_view<char> local_A = A(irreps_A);
                                varray_view<char> local_B = B(irreps_B);

                                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                                auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
                                auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
                                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                                mult(type, comm, cfg, len_AB, len_AC, len_BC, len_ABC,
                                          alpha,       conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, stride_A_AC, stride_A_ABC,
                                                       conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, stride_B_BC, stride_B_ABC,
                                     local_beta, local_conj_C, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, stride_C_BC, stride_C_ABC);

                                local_beta = 1;
                                local_conj_C = false;
                            }
                        }

                        if (local_beta.is_zero())
                        {
                            set(type, comm, cfg, local_C.lengths(),
                                local_beta, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
                        }
                        else if (!local_beta.is_one() || (local_beta.is_complex() && local_conj_C))
                        {
                            scale(type, comm, cfg, local_C.lengths(),
                                  local_beta, local_conj_C, C.data() + (local_C.data()-C.data())*ts, local_C.strides());
                        }
                    }
                }
            }
        }
    }
}

void mult_vec(type_t type, const communicator& comm, const config& cfg,
              const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
              dim_vector idx_A_ABC,
                                   bool conj_B, const dpd_varray_view<char>& B,
              dim_vector idx_B_ABC,
              const scalar&  beta, bool conj_C, const dpd_varray_view<char>& C,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != B.irrep() || A.irrep() != C.irrep())
    {
        if (beta.is_zero())
        {
            set(type, comm, cfg, beta, C, idx_C_ABC);
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            scale(type, comm, cfg, beta, conj_C, C, idx_C_ABC);
        }

        return;
    }

    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    const auto irrep_ABC = C.irrep();

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());

    while (it_ABC.next())
    {
        for (auto i : range(1,idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        varray_view<char> local_A = A(irreps_A);
        varray_view<char> local_B = B(irreps_B);
        varray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);

        mult(type, comm, cfg, {}, {}, {}, len_ABC,
             alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, {}, stride_A_ABC,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, {}, {}, stride_B_ABC,
              beta, conj_C, C.data() + (local_C.data()-C.data())*ts, {}, {}, stride_C_ABC);
    }
}

void mult_vec(type_t type, const communicator& comm, const config& cfg,
              const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
              dim_vector idx_A_AB,
              dim_vector idx_A_ABC,
                                   bool conj_B, const dpd_varray_view<char>& B,
              dim_vector idx_B_AB,
              dim_vector idx_B_ABC,
              const scalar&  beta, bool conj_C, const dpd_varray_view<char>& C,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != B.irrep())
    {
        if (beta.is_zero())
        {
            set(type, comm, cfg, beta, C, idx_C_ABC);
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            scale(type, comm, cfg, beta, conj_C, C, idx_C_ABC);
        }

        return;
    }

    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    auto irrep_ABC = C.irrep();
    auto irrep_AB = A.irrep()^irrep_ABC;

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator it_AB(irrep_AB, nirrep, idx_A_ABC.size());

    while (it_ABC.next())
    {
        for (auto i : range(1,idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        varray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);

        auto local_beta = beta;
        bool local_conj_C = conj_C;

        while (it_AB.next())
        {
            for (auto i : range(idx_A_AB.size()))
            {
                irreps_A[idx_A_AB[i]] =
                irreps_B[idx_B_AB[i]] = it_AB.irrep(i);
            }

            varray_view<char> local_A = A(irreps_A);
            varray_view<char> local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
            auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            mult(type, comm, cfg, len_AB, {}, {}, len_ABC,
                      alpha,       conj_A, A.data() + (local_A.data()-A.data())*ts, stride_A_AB, {}, stride_A_ABC,
                                   conj_B, B.data() + (local_B.data()-B.data())*ts, stride_B_AB, {}, stride_B_ABC,
                 local_beta, local_conj_C, C.data() + (local_C.data()-C.data())*ts, {}, {}, stride_C_ABC);

            local_beta = 1;
            local_conj_C = false;
        }
    }
}

void mult_vec(type_t type, const communicator& comm, const config& cfg,
              const scalar& alpha, bool conj_A, const dpd_varray_view<char>& A,
              dim_vector idx_A_AC,
              dim_vector idx_A_ABC,
                                   bool conj_B, const dpd_varray_view<char>& B,
              dim_vector idx_B_ABC,
              const scalar& beta,  bool conj_C, const dpd_varray_view<char>& C,
              dim_vector idx_C_AC,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != C.irrep())
    {
        if (beta.is_zero())
        {
            set(type, comm, cfg, beta, C, idx_C_ABC);
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            scale(type, comm, cfg, beta, conj_C, C, idx_C_ABC);
        }

        return;
    }

    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    auto irrep_ABC = B.irrep();
    auto irrep_AC = A.irrep()^irrep_ABC;

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator it_AC(irrep_AC, nirrep, idx_A_AC.size());

    while (it_ABC.next())
    while (it_AC.next())
    {
        for (auto i : range(1,idx_A_ABC.size()))
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        for (auto i : range(1,idx_A_AC.size()))
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = it_AC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        varray_view<char> local_A = A(irreps_A);
        varray_view<char> local_B = B(irreps_B);
        varray_view<char> local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_A.lengths(), idx_A_AC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
        auto stride_C_AC = stl_ext::select_from(local_B.strides(), idx_C_AC);

        mult(type, comm, cfg, {}, len_AC, {}, len_ABC,
             alpha, conj_A, A.data() + (local_A.data()-A.data())*ts, {}, stride_A_AC, stride_A_ABC,
                    conj_B, B.data() + (local_B.data()-B.data())*ts, {}, {}, stride_B_ABC,
              beta, conj_C, C.data() + (local_C.data()-C.data())*ts, stride_C_AC, {}, stride_C_ABC);
    }
}

void mult(type_t type, const communicator& comm, const config& cfg,
          const scalar& alpha,
          bool conj_A, const dpd_varray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
          bool conj_B, const dpd_varray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta,
          bool conj_C, const dpd_varray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    if (dpd_impl == FULL)
    {
        switch (type)
        {
            case TYPE_FLOAT:
                mult_full(comm, cfg,
                          alpha.get<float>(), conj_A, reinterpret_cast<const dpd_varray_view<float>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                                              conj_B, reinterpret_cast<const dpd_varray_view<float>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                           beta.get<float>(), conj_C, reinterpret_cast<const dpd_varray_view<float>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
            case TYPE_DOUBLE:
                mult_full(comm, cfg,
                          alpha.get<double>(), conj_A, reinterpret_cast<const dpd_varray_view<double>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                                               conj_B, reinterpret_cast<const dpd_varray_view<double>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                           beta.get<double>(), conj_C, reinterpret_cast<const dpd_varray_view<double>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
            case TYPE_SCOMPLEX:
                mult_full(comm, cfg,
                          alpha.get<scomplex>(), conj_A, reinterpret_cast<const dpd_varray_view<scomplex>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                                                 conj_B, reinterpret_cast<const dpd_varray_view<scomplex>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                           beta.get<scomplex>(), conj_C, reinterpret_cast<const dpd_varray_view<scomplex>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
            case TYPE_DCOMPLEX:
                mult_full(comm, cfg,
                          alpha.get<dcomplex>(), conj_A, reinterpret_cast<const dpd_varray_view<dcomplex>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                                                 conj_B, reinterpret_cast<const dpd_varray_view<dcomplex>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                           beta.get<dcomplex>(), conj_C, reinterpret_cast<const dpd_varray_view<dcomplex>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
        }

        comm.barrier();
        return;
    }
    else if (dpd_impl == BLOCKED)
    {
        mult_block(type, comm, cfg,
                   alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                    beta, conj_C, C, idx_C_AC, idx_C_BC, idx_C_ABC);

        comm.barrier();
        return;
    }

    enum
    {
        HAS_NONE = 0x0,
        HAS_AB   = 0x1,
        HAS_AC   = 0x2,
        HAS_BC   = 0x4,
        HAS_ABC  = 0x8
    };

    int groups = (idx_A_AB.size()  == 0 ? 0 : HAS_AB ) +
                 (idx_A_AC.size()  == 0 ? 0 : HAS_AC ) +
                 (idx_B_BC.size()  == 0 ? 0 : HAS_BC ) +
                 (idx_A_ABC.size() == 0 ? 0 : HAS_ABC);

    switch (groups)
    {
        case HAS_NONE:
        {
            if (comm.master())
                mult(type, alpha, conj_A, A.data(),
                                  conj_B, B.data(),
                            beta, conj_C, C.data());
        }
        break;
        case HAS_AB:
        {
            scalar sum(0, type);

            dot(type, comm, cfg, conj_A, A, idx_A_AB,
                                 conj_B, B, idx_B_AB, sum.raw());

            add(type, alpha, false, sum.raw(), beta, conj_C, C.data());
        }
        break;
        case HAS_AC:
        {
            scalar zero(0, type);
            scalar alpha_B(0, type);

            add(type, alpha, conj_B, B.data(), zero, false, alpha_B.raw());

            add(type, comm, cfg, alpha_B,
                conj_A, A, {}, idx_A_AC, beta, conj_C, C, {}, idx_C_AC);
        }
        break;
        case HAS_BC:
        {
            scalar zero(0, type);
            scalar alpha_A(0, type);

            add(type, alpha, conj_A, A.data(), zero, false, alpha_A.raw());

            add(type, comm, cfg, alpha_A,
                conj_B, B, {}, idx_B_BC, beta, conj_C, C, {}, idx_C_BC);
        }
        break;
        case HAS_ABC:
        {
            mult_vec(type, comm, cfg,
                     alpha, conj_A, A, idx_A_ABC,
                            conj_B, B, idx_B_ABC,
                      beta, conj_C, C, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_A, A, idx_A_AC,
                             conj_B, B, idx_B_BC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC);
        }
        break;
        case HAS_AB+HAS_AC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC,
                             conj_B, B, idx_B_AB,
                       beta, conj_C, C, idx_C_AC);
        }
        break;
        case HAS_AB+HAS_BC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_B, B, idx_B_AB, idx_B_BC,
                             conj_A, A, idx_A_AB,
                       beta, conj_C, C, idx_C_BC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC,
                             conj_B, B, idx_B_AB, idx_B_BC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC);
        }
        break;
        case HAS_AB+HAS_ABC:
        {
            mult_vec(type, comm, cfg,
                     alpha, conj_A, A, idx_A_AB, idx_A_ABC,
                            conj_B, B, idx_B_AB, idx_B_ABC,
                      beta, conj_C, C, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_ABC:
        {
            mult_vec(type, comm, cfg,
                     alpha, conj_A, A, idx_A_AC, idx_A_ABC,
                            conj_B, B, idx_B_ABC,
                      beta, conj_C, C, idx_C_AC, idx_C_ABC);
        }
        break;
        case HAS_BC+HAS_ABC:
        {
            mult_vec(type, comm, cfg,
                     alpha, conj_B, B, idx_B_BC, idx_B_ABC,
                            conj_A, A, idx_A_ABC,
                      beta, conj_C, C, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_A, A, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_BC, idx_B_ABC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_AB, idx_B_ABC,
                       beta, conj_C, C, idx_C_AC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                             conj_A, A, idx_A_AB, idx_A_ABC,
                       beta, conj_C, C, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC, idx_C_ABC);
        }
        break;
    }

    comm.barrier();
}

}
}
