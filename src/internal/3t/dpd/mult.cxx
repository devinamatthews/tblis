#include "mult.hpp"
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

#include "matrix/tensor_matrix.hpp"

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
               T alpha, bool conj_A, const dpd_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        bool conj_B, const dpd_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
               T  beta, bool conj_C, const dpd_varray_view<      T>& C,
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

        mult(comm, cfg, len_AB, len_AC, len_BC, len_ABC,
             alpha, conj_A, A2.data(), stride_A_AB, stride_A_AC, stride_A_ABC,
                    conj_B, B2.data(), stride_B_AB, stride_B_BC, stride_B_ABC,
              beta, conj_C, C2.data(), stride_C_AC, stride_C_BC, stride_C_ABC);

        full_to_block(comm, cfg, C2, C);
    },
    A2, B2, C2);
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, dpd_varray_view<const T> A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
                        bool conj_B, dpd_varray_view<const T> B,
               dim_vector idx_B_AB,
               T beta_, bool conj_C_, dpd_varray_view<      T> C,
               dim_vector idx_C_AC)
{
    unsigned nirrep = A.num_irreps();

    unsigned irrep_AC = C.irrep();
    unsigned irrep_AB = A.irrep()^irrep_AC;

    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_A_AC.size());
    irrep_iterator irrep_it_AB(irrep_AB, nirrep, idx_A_AB.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_AC.next())
    {
        for (unsigned i = 0;i < idx_A_AC.size();i++)
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        auto local_C = C(irreps_C);

        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);

        T beta = beta_;
        bool conj_C = conj_C_;

        if (irrep_AB == B.irrep())
        {
            while (irrep_it_AB.next())
            {
                for (unsigned i = 0;i < idx_A_AB.size();i++)
                {
                    irreps_A[idx_A_AB[i]] =
                    irreps_B[idx_B_AB[i]] = irrep_it_AB.irrep(i);
                }

                auto local_A = A(irreps_A);
                auto local_B = B(irreps_B);

                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

                mult(comm, cfg, len_AB, len_AC, {}, {},
                     alpha, conj_A, local_A.data(), stride_A_AB, stride_A_AC, {},
                            conj_B, local_B.data(), stride_B_AB, {}, {},
                      beta, conj_C, local_C.data(), stride_C_AC, {}, {});

                beta = T(1);
                conj_C = false;
            }
        }

        if (beta == T(0))
        {
            set(comm, cfg, local_C.lengths(), T(0), local_C.data(), local_C.strides());
        }
        else if (beta != T(1) || (is_complex<T>::value && conj_C))
        {
            scale(comm, cfg, local_C.lengths(), beta, conj_C, local_C.data(), local_C.strides());
        }
    }
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, dpd_varray_view<const T> A,
               dim_vector idx_A_AC,
                        bool conj_B, dpd_varray_view<const T> B,
               dim_vector idx_B_BC,
               T beta,  bool conj_C, dpd_varray_view<      T> C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC)
{
    unsigned nirrep = A.num_irreps();

    for (unsigned irrep_AC = 0;irrep_AC < nirrep;irrep_AC++)
    {
        unsigned irrep_BC = C.irrep()^irrep_AC;

        irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_C_AC.size());
        irrep_iterator irrep_it_BC(irrep_BC, nirrep, idx_C_BC.size());

        irrep_vector irreps_A(A.dimension());
        irrep_vector irreps_B(B.dimension());
        irrep_vector irreps_C(C.dimension());

        while (irrep_it_AC.next())
        while (irrep_it_BC.next())
        {
            for (unsigned i = 0;i < idx_A_AC.size();i++)
            {
                irreps_A[idx_A_AC[i]] =
                irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
            }

            for (unsigned i = 0;i < idx_B_BC.size();i++)
            {
                irreps_B[idx_B_BC[i]] =
                irreps_C[idx_C_BC[i]] = irrep_it_BC.irrep(i);
            }

            auto local_C = C(irreps_C);

            auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
            auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
            auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
            auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

            if (irrep_AC == A.irrep() && irrep_BC == B.irrep())
            {
                auto local_A = A(irreps_A);
                auto local_B = B(irreps_B);

                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                mult(comm, cfg, {}, len_AC, len_BC, {},
                     alpha, conj_A, local_A.data(), {}, stride_A_AC, {},
                            conj_B, local_B.data(), {}, stride_B_BC, {},
                      beta, conj_C, local_C.data(), stride_C_AC, stride_C_BC, {});
            }
            else if (beta == T(0))
            {
                set(comm, cfg, local_C.lengths(), T(0), local_C.data(), local_C.strides());
            }
            else if (beta != T(1) || (is_complex<T>::value && conj_C))
            {
                scale(comm, cfg, local_C.lengths(), beta, conj_C, local_C.data(), local_C.strides());
            }
        }
    }
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, dpd_varray_view<const T> A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
                        bool conj_B, dpd_varray_view<const T> B,
               dim_vector idx_B_AB,
               dim_vector idx_B_BC,
               T beta,  bool conj_C, dpd_varray_view<      T> C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC)
{
    if ((A.irrep()^B.irrep()) != C.irrep())
    {
        if (beta == T(0))
        {
            set(comm, cfg, T(0), C, {});
        }
        else if (beta != T(1) || (is_complex<T>::value && conj_C))
        {
            scale(comm, cfg, beta, conj_C, C, {});
        }

        return;
    }

    //TODO
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    unsigned nirrep = A.num_irreps();

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

    unsigned unit_A_AC = unit_dim(stride[0], idx_A_AC);
    unsigned unit_C_AC = unit_dim(stride[2], idx_C_AC);
    unsigned unit_B_BC = unit_dim(stride[1], idx_B_BC);
    unsigned unit_C_BC = unit_dim(stride[2], idx_C_BC);
    unsigned unit_A_AB = unit_dim(stride[0], idx_A_AB);
    unsigned unit_B_AB = unit_dim(stride[1], idx_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == perm_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == perm_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == perm_AB.size() && unit_B_AB == perm_AB.size()));

    bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < perm_AC.size();
    bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < perm_BC.size();
    bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < perm_AB.size()) ||
                     (unit_B_AB > 0 && unit_B_AB < perm_AB.size());

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

    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_AC = A.irrep()^irrep_AB;
        unsigned irrep_BC = B.irrep()^irrep_AB;

        dpd_tensor_matrix<T> at(A, idx_A_AC, idx_A_AB, irrep_AB, pack_M_3d, pack_K_3d);
        dpd_tensor_matrix<T> bt(B, idx_B_AB, idx_B_BC, irrep_BC, pack_K_3d, pack_N_3d);
        dpd_tensor_matrix<T> ct(C, idx_C_AC, idx_C_BC, irrep_BC, pack_M_3d, pack_N_3d);

        if (ct.length(0) == 0 || ct.length(1) == 0) continue;

        if (at.length(1) != 0)
        {
            TensorGEMM{}(comm, cfg, alpha, at, bt, beta, ct);
        }
        else if (beta != T(1))
        {
            irrep_iterator row_it(irrep_AC, nirrep, idx_C_AC.size());
            irrep_iterator col_it(irrep_BC, nirrep, idx_C_BC.size());

            irrep_vector irreps(C.dimension());

            while (row_it.next())
            {
                for (unsigned i = 0;i < idx_C_AC.size();i++)
                    irreps[idx_C_AC[i]] = row_it.irrep(i);

                while (col_it.next())
                {
                    for (unsigned i = 0;i < idx_C_BC.size();i++)
                        irreps[idx_C_BC[i]] = col_it.irrep(i);

                    auto local_C = C(irreps);

                    if (beta == T(0))
                    {
                        set(comm, cfg, local_C.lengths(),
                            beta, local_C.data(), local_C.strides());
                    }
                    else
                    {
                        scale(comm, cfg, local_C.lengths(),
                              beta, false, local_C.data(), local_C.strides());
                    }
                }
            }
        }
    }
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, dpd_varray_view<const T> A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                        bool conj_B, dpd_varray_view<const T> B,
               dim_vector idx_B_AB,
               dim_vector idx_B_ABC,
               T beta_, bool conj_C_, dpd_varray_view<      T> C,
               dim_vector idx_C_AC,
               dim_vector idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();

    unsigned irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    unsigned irrep_AB = A.irrep()^C.irrep();
    unsigned irrep_AC = A.irrep()^B.irrep();

    irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator irrep_it_AC(irrep_AC, nirrep, idx_A_AC.size());
    irrep_iterator irrep_it_AB(irrep_AB, nirrep, idx_A_AB.size());

    irrep_vector irreps_A(A.dimension());
    irrep_vector irreps_B(B.dimension());
    irrep_vector irreps_C(C.dimension());

    while (irrep_it_ABC.next())
    while (irrep_it_AC.next())
    {
        for (unsigned i = 0;i < idx_A_ABC.size();i++)
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = irrep_it_ABC.irrep(i);
        }

        for (unsigned i = 0;i < idx_A_AC.size();i++)
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        auto local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);

        T beta = beta_;
        bool conj_C = conj_C_;

        while (irrep_it_AB.next())
        {
            for (unsigned i = 0;i < idx_A_AB.size();i++)
            {
                irreps_A[idx_A_AB[i]] =
                irreps_B[idx_B_AB[i]] = irrep_it_AB.irrep(i);
            }

            auto local_A = A(irreps_A);
            auto local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
            auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
            auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            mult(comm, cfg, len_AB, len_AC, {}, len_ABC,
                 alpha, conj_A, local_A.data(), stride_A_AB, stride_A_AC, stride_A_ABC,
                        conj_B, local_B.data(), stride_B_AB, {}, stride_B_ABC,
                  beta, conj_C, local_C.data(), stride_C_AC, {}, stride_C_ABC);

            beta = T(1);
            conj_C = false;
        }
    }
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, dpd_varray_view<const T> A,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                        bool conj_B, dpd_varray_view<const T> B,
               dim_vector idx_B_BC,
               dim_vector idx_B_ABC,
               T beta,  bool conj_C, dpd_varray_view<      T> C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC,
               dim_vector idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();

    irrep_vector irreps_ABC(idx_A_ABC.size());
    len_vector len_ABC(idx_A_ABC.size());

    unsigned irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    unsigned irrep_AC = A.irrep()^irrep_ABC;
    unsigned irrep_BC = B.irrep()^irrep_ABC;

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
        for (unsigned i = 0;i < idx_A_ABC.size();i++)
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = irrep_it_ABC.irrep(i);
        }

        for (unsigned i = 0;i < idx_A_AC.size();i++)
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = irrep_it_AC.irrep(i);
        }

        for (unsigned i = 0;i < idx_B_BC.size();i++)
        {
            irreps_B[idx_B_BC[i]] =
            irreps_C[idx_C_BC[i]] = irrep_it_BC.irrep(i);
        }

        auto local_A = A(irreps_A);
        auto local_B = B(irreps_B);
        auto local_C = C(irreps_C);

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

        mult(comm, cfg, {}, len_AC, len_BC, len_ABC,
             alpha, conj_A, local_A.data(), {}, stride_A_AC, stride_A_ABC,
                    conj_B, local_B.data(), {}, stride_B_BC, stride_B_ABC,
              beta, conj_C, local_C.data(), stride_C_AC, stride_C_BC, stride_C_ABC);
    }
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, dpd_varray_view<const T> A,
               dim_vector idx_A_AB,
               dim_vector idx_A_AC,
               dim_vector idx_A_ABC,
                        bool conj_B, dpd_varray_view<const T> B,
               dim_vector idx_B_AB,
               dim_vector idx_B_BC,
               dim_vector idx_B_ABC,
               T beta,  bool conj_C, dpd_varray_view<      T> C,
               dim_vector idx_C_AC,
               dim_vector idx_C_BC,
               dim_vector idx_C_ABC)
{
    //TODO
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    unsigned nirrep = A.num_irreps();

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

    unsigned unit_A_AC = unit_dim(stride[0], idx_A_AC);
    unsigned unit_C_AC = unit_dim(stride[2], idx_C_AC);
    unsigned unit_B_BC = unit_dim(stride[1], idx_B_BC);
    unsigned unit_C_BC = unit_dim(stride[2], idx_C_BC);
    unsigned unit_A_AB = unit_dim(stride[0], idx_A_AB);
    unsigned unit_B_AB = unit_dim(stride[1], idx_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == perm_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == perm_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == perm_AB.size() && unit_B_AB == perm_AB.size()));

    bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < perm_AC.size();
    bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < perm_BC.size();
    bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < perm_AB.size()) ||
                     (unit_B_AB > 0 && unit_B_AB < perm_AB.size());

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

    irrep_vector irreps_ABC(idx_A_ABC.size());
    len_vector len_ABC(idx_A_ABC.size());

    for (unsigned irrep_ABC = 0;irrep_ABC < nirrep;irrep_ABC++)
    {
        for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
        {
            unsigned irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
            unsigned irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

            irrep_iterator irrep_it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());

            while (irrep_it_ABC.next())
            {
                for (unsigned i = 0;i < idx_A_ABC.size();i++)
                {
                    irreps_ABC[i] = irrep_it_ABC.irrep(i);
                    len_ABC[i] = A.length(idx_A_ABC[i], irreps_ABC[i]);
                }

                viterator<0> it_ABC(len_ABC);

                while (it_ABC.next())
                {
                    dpd_tensor_matrix<T> at(A, idx_A_AC, idx_A_AB, irrep_AB, idx_A_ABC,
                                            irreps_ABC, it_ABC.position(), pack_M_3d, pack_K_3d);
                    dpd_tensor_matrix<T> bt(B, idx_B_AB, idx_B_BC, irrep_BC, idx_B_ABC,
                                            irreps_ABC, it_ABC.position(), pack_K_3d, pack_N_3d);
                    dpd_tensor_matrix<T> ct(C, idx_C_AC, idx_C_BC, irrep_BC, idx_C_ABC,
                                            irreps_ABC, it_ABC.position(), pack_M_3d, pack_N_3d);

                    if (ct.length(0) == 0 || ct.length(1) == 0) continue;

                    if (at.length(1) != 0)
                    {
                        TensorGEMM{}(comm, cfg, alpha, at, bt, beta, ct);
                    }
                    else if (beta != T(1))
                    {
                        irrep_iterator row_it(irrep_AC, nirrep, idx_C_AC.size());
                        irrep_iterator col_it(irrep_BC, nirrep, idx_C_BC.size());

                        irrep_vector irreps(C.dimension());

                        while (row_it.next())
                        {
                            for (unsigned i = 0;i < idx_C_AC.size();i++)
                                irreps[idx_C_AC[i]] = row_it.irrep(i);

                            while (col_it.next())
                            {
                                for (unsigned i = 0;i < idx_C_BC.size();i++)
                                    irreps[idx_C_BC[i]] = col_it.irrep(i);

                                auto local_C = C(irreps);

                                if (beta == T(0))
                                {
                                    set(comm, cfg, local_C.lengths(),
                                        beta, local_C.data(), local_C.strides());
                                }
                                else
                                {
                                    scale(comm, cfg, local_C.lengths(),
                                          beta, false, local_C.data(), local_C.strides());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void mult_block(const communicator& comm, const config& cfg,
                T alpha, bool conj_A, dpd_varray_view<const T> A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                         bool conj_B, dpd_varray_view<const T> B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                T beta_, bool conj_C_, dpd_varray_view<      T> C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();
    unsigned ndim_C = C.dimension();

    unsigned ndim_AC = idx_C_AC.size();
    unsigned ndim_BC = idx_C_BC.size();
    unsigned ndim_AB = idx_A_AB.size();
    unsigned ndim_ABC = idx_A_ABC.size();

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

    stride_type dense_AC = 1;
    stride_type nblock_AC = 1;
    for (unsigned i : idx_C_AC)
    {
        dense_AC *= len[2][i];
        nblock_AC *= nirrep;
    }
    dense_AC /= nblock_AC;

    stride_type dense_BC = 1;
    stride_type nblock_BC = 1;
    for (unsigned i : idx_C_BC)
    {
        dense_BC *= len[2][i];
        nblock_BC *= nirrep;
    }
    dense_BC /= nblock_BC;

    stride_type dense_AB = 1;
    stride_type nblock_AB = 1;
    for (unsigned i : idx_A_AB)
    {
        dense_AB *= len[0][i];
        nblock_AB *= nirrep;
    }
    dense_AB /= nblock_AB;

    stride_type dense_ABC = 1;
    stride_type nblock_ABC = 1;
    for (unsigned i : idx_A_ABC)
    {
        dense_ABC *= len[0][i];
        nblock_ABC *= nirrep;
    }
    dense_ABC /= nblock_ABC;

    if (nblock_AC > 1) nblock_AC /= nirrep;
    if (nblock_BC > 1) nblock_BC /= nirrep;
    if (nblock_AB > 1) nblock_AB /= nirrep;
    if (nblock_ABC > 1) nblock_ABC /= nirrep;

    irrep_vector irreps_A(ndim_A);
    irrep_vector irreps_B(ndim_B);
    irrep_vector irreps_C(ndim_C);

    for (unsigned irrep_ABC = 0;irrep_ABC < nirrep;irrep_ABC++)
    {
        if (ndim_ABC == 0 && irrep_ABC != 0) continue;

        for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
        {
            unsigned irrep_AC = A.irrep()^irrep_ABC^irrep_AB;
            unsigned irrep_BC = C.irrep()^irrep_ABC^irrep_AC;

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

                        auto local_C = C(irreps_C);

                        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
                        auto len_AC = stl_ext::select_from(local_C.lengths(), idx_C_AC);
                        auto len_BC = stl_ext::select_from(local_C.lengths(), idx_C_BC);
                        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
                        auto stride_C_AC = stl_ext::select_from(local_C.strides(), idx_C_AC);
                        auto stride_C_BC = stl_ext::select_from(local_C.strides(), idx_C_BC);

                        T beta = beta_;
                        bool conj_C = conj_C_;

                        if ((ndim_AB != 0 || irrep_AB == 0) &&
                            irrep_ABC == (A.irrep()^B.irrep()^C.irrep()))
                        {
                            for (stride_type block_AB = 0;block_AB < nblock_AB;block_AB++)
                            {
                                assign_irreps(ndim_AB, irrep_AB, nirrep, block_AB,
                                              irreps_A, idx_A_AB, irreps_B, idx_B_AB);

                                if (is_block_empty(A, irreps_A)) continue;

                                auto local_A = A(irreps_A);
                                auto local_B = B(irreps_B);

                                auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
                                auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
                                auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
                                auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
                                auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);
                                auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
                                auto stride_B_BC = stl_ext::select_from(local_B.strides(), idx_B_BC);

                                mult(comm, cfg, len_AB, len_AC, len_BC, len_ABC,
                                     alpha, conj_A, local_A.data(), stride_A_AB, stride_A_AC, stride_A_ABC,
                                            conj_B, local_B.data(), stride_B_AB, stride_B_BC, stride_B_ABC,
                                      beta, conj_C, local_C.data(), stride_C_AC, stride_C_BC, stride_C_ABC);

                                beta = T(1);
                                conj_C = false;
                            }
                        }

                        if (beta == T(0))
                        {
                            set(comm, cfg, local_C.lengths(),
                                beta, local_C.data(), local_C.strides());
                        }
                        else if (beta != T(1) || (is_complex<T>::value && conj_C))
                        {
                            scale(comm, cfg, local_C.lengths(),
                                  beta, conj_C, local_C.data(), local_C.strides());
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void mult_vec(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, dpd_varray_view<const T> A,
              dim_vector idx_A_ABC,
                       bool conj_B, dpd_varray_view<const T> B,
              dim_vector idx_B_ABC,
              T  beta, bool conj_C, dpd_varray_view<      T> C,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != B.irrep() || A.irrep() != C.irrep())
    {
        if (beta == T(0))
        {
            set(comm, cfg, T(0), C, idx_C_ABC);
        }
        else if (beta != T(1) || (is_complex<T>::value && conj_C))
        {
            scale(comm, cfg, beta, conj_C, C, idx_C_ABC);
        }

        return;
    }

    unsigned nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    unsigned irrep_ABC = C.irrep();

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());

    while (it_ABC.next())
    {
        for (unsigned i = 0;i < idx_A_ABC.size();i++)
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        auto local_A = A(irreps_A);
        auto local_B = B(irreps_B);
        auto local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);

        mult(comm, cfg, {}, {}, {}, len_ABC,
             alpha, conj_A, local_A.data(), {}, {}, stride_A_ABC,
                    conj_B, local_B.data(), {}, {}, stride_B_ABC,
              beta, conj_C, local_C.data(), {}, {}, stride_C_ABC);
    }
}

template <typename T>
void mult_vec(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, dpd_varray_view<const T> A,
              dim_vector idx_A_AB,
              dim_vector idx_A_ABC,
                       bool conj_B, dpd_varray_view<const T> B,
              dim_vector idx_B_AB,
              dim_vector idx_B_ABC,
              T beta_, bool conj_C_, dpd_varray_view<      T> C,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != B.irrep())
    {
        if (beta_ == T(0))
        {
            set(comm, cfg, T(0), C, idx_C_ABC);
        }
        else if (beta_ != T(1) || (is_complex<T>::value && conj_C_))
        {
            scale(comm, cfg, beta_, conj_C_, C, idx_C_ABC);
        }

        return;
    }

    unsigned nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    unsigned irrep_ABC = C.irrep();
    unsigned irrep_AB = A.irrep()^irrep_ABC;

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator it_AB(irrep_AB, nirrep, idx_A_ABC.size());

    while (it_ABC.next())
    {
        for (unsigned i = 0;i < idx_A_ABC.size();i++)
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        auto local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);

        T beta = beta_;
        bool conj_C = conj_C_;

        while (it_AB.next())
        {
            for (unsigned i = 0;i < idx_A_AB.size();i++)
            {
                irreps_A[idx_A_AB[i]] =
                irreps_B[idx_B_AB[i]] = it_AB.irrep(i);
            }

            auto local_A = A(irreps_A);
            auto local_B = B(irreps_B);

            auto len_AB = stl_ext::select_from(local_A.lengths(), idx_A_AB);
            auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
            auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
            auto stride_A_AB = stl_ext::select_from(local_A.strides(), idx_A_AB);
            auto stride_B_AB = stl_ext::select_from(local_B.strides(), idx_B_AB);

            mult(comm, cfg, len_AB, {}, {}, len_ABC,
                 alpha, conj_A, local_A.data(), stride_A_AB, {}, stride_A_ABC,
                        conj_B, local_B.data(), stride_B_AB, {}, stride_B_ABC,
                  beta, conj_C, local_C.data(), {}, {}, stride_C_ABC);

            beta = T(1);
            conj_C = false;
        }
    }
}

template <typename T>
void mult_vec(const communicator& comm, const config& cfg,
              T alpha, bool conj_A, dpd_varray_view<const T> A,
              dim_vector idx_A_AC,
              dim_vector idx_A_ABC,
                       bool conj_B, dpd_varray_view<const T> B,
              dim_vector idx_B_ABC,
              T beta,  bool conj_C, dpd_varray_view<      T> C,
              dim_vector idx_C_AC,
              dim_vector idx_C_ABC)
{
    if (A.irrep() != C.irrep())
    {
        if (beta == T(0))
        {
            set(comm, cfg, T(0), C, idx_C_ABC);
        }
        else if (beta != T(1) || (is_complex<T>::value && conj_C))
        {
            scale(comm, cfg, beta, conj_C, C, idx_C_ABC);
        }

        return;
    }

    unsigned nirrep = A.num_irreps();

    irrep_vector irreps_A(idx_A_ABC.size());
    irrep_vector irreps_B(idx_A_ABC.size());
    irrep_vector irreps_C(idx_A_ABC.size());

    unsigned irrep_ABC = B.irrep();
    unsigned irrep_AC = A.irrep()^irrep_ABC;

    irrep_iterator it_ABC(irrep_ABC, nirrep, idx_A_ABC.size());
    irrep_iterator it_AC(irrep_AC, nirrep, idx_A_AC.size());

    while (it_ABC.next())
    while (it_AC.next())
    {
        for (unsigned i = 0;i < idx_A_ABC.size();i++)
        {
            irreps_A[idx_A_ABC[i]] =
            irreps_B[idx_B_ABC[i]] =
            irreps_C[idx_C_ABC[i]] = it_ABC.irrep(i);
        }

        for (unsigned i = 0;i < idx_A_AC.size();i++)
        {
            irreps_A[idx_A_AC[i]] =
            irreps_C[idx_C_AC[i]] = it_AC.irrep(i);
        }

        if (is_block_empty(C, irreps_C)) continue;

        auto local_A = A(irreps_A);
        auto local_B = B(irreps_B);
        auto local_C = C(irreps_C);

        auto len_ABC = stl_ext::select_from(local_C.lengths(), idx_C_ABC);
        auto len_AC = stl_ext::select_from(local_A.lengths(), idx_A_AC);
        auto stride_A_ABC = stl_ext::select_from(local_A.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(local_B.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(local_C.strides(), idx_C_ABC);
        auto stride_A_AC = stl_ext::select_from(local_A.strides(), idx_A_AC);
        auto stride_C_AC = stl_ext::select_from(local_B.strides(), idx_C_AC);

        mult(comm, cfg, {}, len_AC, {}, len_ABC,
             alpha, conj_A, local_A.data(), {}, stride_A_AC, stride_A_ABC,
                    conj_B, local_B.data(), {}, {}, stride_B_ABC,
              beta, conj_C, local_C.data(), stride_C_AC, {}, stride_C_ABC);
    }
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          T alpha, bool conj_A, const dpd_varray_view<const T>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                   bool conj_B, const dpd_varray_view<const T>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          T  beta, bool conj_C, const dpd_varray_view<      T>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    if (dpd_impl == FULL)
    {
        mult_full(comm, cfg,
                  alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                         conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                   beta, conj_C, C, idx_C_AC, idx_C_BC, idx_C_ABC);

        comm.barrier();
        return;
    }
    else if (dpd_impl == BLOCKED)
    {
        mult_block(comm, cfg,
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
            {
                if (beta == T(0))
                {
                    *C.data() = alpha*(conj_A ? conj(*A.data()) : *A.data())*
                                      (conj_B ? conj(*B.data()) : *B.data());
                }
                else
                {
                    *C.data() = alpha*(conj_A ? conj(*A.data()) : *A.data())*
                                      (conj_B ? conj(*B.data()) : *B.data()) +
                                 beta*(conj_C ? conj(*C.data()) : *C.data());
                }
            }
        }
        break;
        case HAS_AB:
        {
            T sum = 0;
            dot(comm, cfg, conj_A, A, idx_A_AB,
                           conj_B, B, idx_B_AB, sum);

            if (comm.master())
            {
                if (beta == T(0))
                {
                    *C.data() = alpha*sum;
                }
                else
                {
                    *C.data() = alpha*sum + beta*(conj_C ? conj(*C.data()) : *C.data());
                }
            }
        }
        break;
        case HAS_AC:
        {
            add(comm, cfg, alpha*(conj_B ? conj(*B.data()) : *B.data()),
                conj_A, A, {}, idx_A_AC, beta, conj_C, C, {}, idx_C_AC);
        }
        break;
        case HAS_BC:
        {
            add(comm, cfg, alpha*(conj_A ? conj(*A.data()) : *A.data()),
                conj_B, B, {}, idx_B_BC, beta, conj_C, C, {}, idx_C_BC);
        }
        break;
        case HAS_ABC:
        {
            mult_vec(comm, cfg,
                     alpha, conj_A, A, idx_A_ABC,
                            conj_B, B, idx_B_ABC,
                      beta, conj_C, C, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_A, A, idx_A_AC,
                             conj_B, B, idx_B_BC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC);
        }
        break;
        case HAS_AB+HAS_AC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC,
                             conj_B, B, idx_B_AB,
                       beta, conj_C, C, idx_C_AC);
        }
        break;
        case HAS_AB+HAS_BC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_B, B, idx_B_AB, idx_B_BC,
                             conj_A, A, idx_A_AB,
                       beta, conj_C, C, idx_C_BC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC,
                             conj_B, B, idx_B_AB, idx_B_BC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC);
        }
        break;
        case HAS_AB+HAS_ABC:
        {
            mult_vec(comm, cfg,
                     alpha, conj_A, A, idx_A_AB, idx_A_ABC,
                            conj_B, B, idx_B_AB, idx_B_ABC,
                      beta, conj_C, C, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_ABC:
        {
            mult_vec(comm, cfg,
                     alpha, conj_A, A, idx_A_AC, idx_A_ABC,
                            conj_B, B, idx_B_ABC,
                      beta, conj_C, C, idx_C_AC, idx_C_ABC);
        }
        break;
        case HAS_BC+HAS_ABC:
        {
            mult_vec(comm, cfg,
                     alpha, conj_B, B, idx_B_BC, idx_B_ABC,
                            conj_A, A, idx_A_ABC,
                      beta, conj_C, C, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_A, A, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_BC, idx_B_ABC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_AB, idx_B_ABC,
                       beta, conj_C, C, idx_C_AC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_BC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                             conj_A, A, idx_A_AB, idx_A_ABC,
                       beta, conj_C, C, idx_C_BC, idx_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                             conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                       beta, conj_C, C, idx_C_AC, idx_C_BC, idx_C_ABC);
        }
        break;
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   T alpha, bool conj_A, const dpd_varray_view<const T>& A, \
                   const dim_vector& idx_A_AB, \
                   const dim_vector& idx_A_AC, \
                   const dim_vector& idx_A_ABC, \
                            bool conj_B, const dpd_varray_view<const T>& B, \
                   const dim_vector& idx_B_AB, \
                   const dim_vector& idx_B_BC, \
                   const dim_vector& idx_B_ABC, \
                   T  beta, bool conj_C, const dpd_varray_view<      T>& C, \
                   const dim_vector& idx_C_AC, \
                   const dim_vector& idx_C_BC, \
                   const dim_vector& idx_C_ABC);
#include "configs/foreach_type.h"

}
}
