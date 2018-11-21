#include "mult.hpp"
#include "internal/1t/indexed_dpd/util.hpp"
#include "internal/1t/indexed_dpd/set.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/3t/dense/mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "matrix/tensor_matrix.hpp"

#include "nodes/gemm.hpp"

#include "external/stl_ext/include/iostream.hpp"

namespace MArray
{

template <typename T, size_t N, typename Allocator>
std::ostream& operator<<(std::ostream& os, const short_vector<T, N, Allocator>& v)
{
    return os << std::vector<T>(v.begin(), v.end());
}

}

namespace tblis
{
namespace internal
{


template <typename T, unsigned N>
std::ostream& operator<<(std::ostream& os, const index_set<T, N>& v)
{
    os << "\n{\n";
    os << "\toffset, factor: " << v.offset << " " << v.factor << '\n';
    for (unsigned i = 0;i < N;i++)
        os << "\tkey, idx: " << v.key[i] << " " << v.idx[i] << '\n';
    return os << '}';
}

template <typename T>
void mult_full(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        bool conj_B, const indexed_dpd_varray_view<const T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
                                     const indexed_dpd_varray_view<      T>& C,
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
              T(0),  false, C2.data(), stride_C_AC, stride_C_BC, stride_C_ABC);

        full_to_block(comm, cfg, C2, C);
    },
    A2, B2, C2);
}

template <typename T, typename U>
void append(T& t, const U& u)
{
    t.insert(t.end(), u.begin(), u.end());
}

template <typename T, typename U>
T appended(T t, const U& u)
{
    append(t, u);
    return t;
}

template <typename T>
void contract_block_nofuse(const communicator& comm, const config& cfg,
                           T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                           dim_vector idx_A_AB,
                           dim_vector idx_A_AC,
                                    bool conj_B, const indexed_dpd_varray_view<const T>& B,
                           dim_vector idx_B_AB,
                           dim_vector idx_B_BC,
                                                 const indexed_dpd_varray_view<      T>& C,
                           dim_vector idx_C_AC,
                           dim_vector idx_C_BC)
{
    unsigned nirrep = A.num_irreps();

    unsigned ndim_AC = idx_C_AC.size();
    unsigned ndim_BC = idx_C_BC.size();
    unsigned ndim_AB = idx_A_AB.size();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(C, idx_C_AC, A, idx_A_AC);
    dpd_index_group<2> group_BC(C, idx_C_BC, B, idx_B_BC);

    for (auto& len : group_AB.batch_len) if (len == 0) return;
    for (auto& len : group_AC.batch_len) if (len == 0) return;
    for (auto& len : group_BC.batch_len) if (len == 0) return;

    group_indices<T, 2> indices_A(A, group_AC, 1, group_AB, 0);
    group_indices<T, 2> indices_B(B, group_BC, 1, group_AB, 1);
    group_indices<T, 2> indices_C(C, group_AC, 0, group_BC, 0);

    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    auto mixed_dim_A = appended(group_AC.mixed_idx[1],
                                group_AB.mixed_idx[0]);
    auto mixed_dim_B = appended(group_AB.mixed_idx[1],
                                group_BC.mixed_idx[1]);
    auto mixed_dim_C = appended(group_AC.mixed_idx[0],
                                group_BC.mixed_idx[0]);

    auto mixed_irrep_A = appended(stl_ext::select_from(group_AC.batch_irrep, group_AC.mixed_pos[1]),
                                  stl_ext::select_from(group_AB.batch_irrep, group_AB.mixed_pos[0]));
    auto mixed_irrep_B = appended(stl_ext::select_from(group_AB.batch_irrep, group_AB.mixed_pos[1]),
                                  stl_ext::select_from(group_BC.batch_irrep, group_BC.mixed_pos[1]));
    auto mixed_irrep_C = appended(stl_ext::select_from(group_AC.batch_irrep, group_AC.mixed_pos[0]),
                                  stl_ext::select_from(group_BC.batch_irrep, group_BC.mixed_pos[0]));

    for (unsigned irrep_AB0 = 0;irrep_AB0 < nirrep;irrep_AB0++)
    {
        unsigned irrep_AB = irrep_AB0;
        unsigned irrep_AC = A.irrep()^irrep_AB0;
        unsigned irrep_BC = B.irrep()^irrep_AB0;

        for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
        for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
        for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

        if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
        if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
        if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

        stride_type idx = 0;
        stride_type idx_A = 0;
        stride_type idx_C = 0;

        comm.do_tasks_deferred(nidx_C,
                               group_AB.dense_size*group_AC.dense_size*group_BC.dense_size/inout_ratio,
        [&](communicator::deferred_task_set& tasks)
        {
            for_each_match<true, true>(idx_A, nidx_A, indices_A, 0,
                                       idx_C, nidx_C, indices_C, 0,
            [&](stride_type next_A, stride_type next_C)
            {
                stride_type idx_B = 0;

                for_each_match<true, false>(idx_B, nidx_B, indices_B, 0,
                                            idx_C, next_C, indices_C, 1,
                [&](stride_type next_B)
                {
                    if (indices_C[idx_C].factor == T(0)) return;

                    tasks.visit(idx++,
                    [&,idx,idx_A,idx_B,idx_C,next_A,next_B]
                    (const communicator& subcomm)
                    {
                        auto local_idx_A = idx_A;
                        auto local_idx_B = idx_B;

                        auto dpd_A = A[0];
                        auto dpd_B = B[0];
                        auto dpd_C = C[0];

                        for_each_match<false, false>(local_idx_A, next_A, indices_A, 1,
                                                     local_idx_B, next_B, indices_B, 1,
                        [&]
                        {
                            auto factor = alpha*indices_A[local_idx_A].factor*
                                                indices_B[local_idx_B].factor*
                                                indices_C[      idx_C].factor;
                            if (factor == T(0)) return;

                            dpd_A.data(A.data(0) + indices_A[local_idx_A].offset);
                            dpd_B.data(B.data(0) + indices_B[local_idx_B].offset);
                            dpd_C.data(C.data(0) + indices_C[      idx_C].offset);

                            auto mixed_idx_A = appended(stl_ext::select_from(indices_A[local_idx_A].idx[0], group_AC.mixed_pos[1]),
                                                        stl_ext::select_from(indices_A[local_idx_A].idx[1], group_AB.mixed_pos[0]));
                            auto mixed_idx_B = appended(stl_ext::select_from(indices_B[local_idx_B].idx[1], group_AB.mixed_pos[1]),
                                                        stl_ext::select_from(indices_B[local_idx_B].idx[0], group_BC.mixed_pos[1]));
                            auto mixed_idx_C = appended(stl_ext::select_from(indices_C[      idx_C].idx[0], group_AC.mixed_pos[0]),
                                                        stl_ext::select_from(indices_C[      idx_C].idx[1], group_BC.mixed_pos[0]));

                            dpd_tensor_matrix<T> at(dpd_A, group_AC.dense_idx[1], group_AB.dense_idx[0], irrep_AB,
                                                    mixed_dim_A, mixed_irrep_A, mixed_idx_A, group_AC.pack_3d, group_AB.pack_3d);
                            dpd_tensor_matrix<T> bt(dpd_B, group_AB.dense_idx[1], group_BC.dense_idx[1], irrep_BC,
                                                    mixed_dim_B, mixed_irrep_B, mixed_idx_B, group_AB.pack_3d, group_BC.pack_3d);
                            dpd_tensor_matrix<T> ct(dpd_C, group_AC.dense_idx[0], group_BC.dense_idx[0], irrep_BC,
                                                    mixed_dim_C, mixed_irrep_C, mixed_idx_C, group_AC.pack_3d, group_BC.pack_3d);

                            TensorGEMM{}(subcomm, cfg, factor, at, bt, T(1), ct);
                        });
                    });
                });
            });
        });
    }
}

template <typename T>
void contract_block_fuse_AB(const communicator& comm, const config& cfg,
                            T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                            dim_vector idx_A_AB,
                            dim_vector idx_A_AC,
                                     bool conj_B, const indexed_dpd_varray_view<const T>& B,
                            dim_vector idx_B_AB,
                            dim_vector idx_B_BC,
                                                  const indexed_dpd_varray_view<      T>& C,
                            dim_vector idx_C_AC,
                            dim_vector idx_C_BC)
{
    //TODO
}

template <typename T>
void mult_block(const communicator& comm, const config& cfg,
                T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                         bool conj_B, const indexed_dpd_varray_view<const T>& B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                                      const indexed_dpd_varray_view<      T>& C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    unsigned nirrep = A.num_irreps();

    unsigned ndim_ABC = idx_C_ABC.size();
    unsigned ndim_AC = idx_C_AC.size();
    unsigned ndim_BC = idx_C_BC.size();
    unsigned ndim_AB = idx_A_AB.size();

    dpd_index_group<3> group_ABC(A, idx_A_ABC, B, idx_B_ABC, C, idx_C_ABC);
    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_ABC, irreps_A, irreps_B, irreps_C);
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<T, 3> indices_A(A, group_ABC, 0, group_AC, 0, group_AB, 0);
    group_indices<T, 3> indices_B(B, group_ABC, 1, group_BC, 0, group_AB, 1);
    group_indices<T, 3> indices_C(C, group_ABC, 2, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B0 = 0;
    stride_type idx_C = 0;

    unsigned irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    for (auto irrep : group_ABC.batch_irrep) irrep_ABC ^= irrep;

    if (group_ABC.dense_ndim == 0 && irrep_ABC != 0) return;

    comm.do_tasks_deferred(nirrep*nidx_C*group_AC.dense_nblock*group_BC.dense_nblock*group_ABC.dense_nblock,
                           group_AB.dense_size*group_AC.dense_size*group_BC.dense_size*group_ABC.dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, true, true>(idx_A,  nidx_A, indices_A, 0,
                                        idx_B0, nidx_B, indices_B, 0,
                                        idx_C,  nidx_C, indices_C, 0,
        [&](stride_type next_A_ABC, stride_type next_B_ABC, stride_type next_C_ABC)
        {
            for_each_match<true, true>(idx_A, next_A_ABC, indices_A, 1,
                                      idx_C, next_C_ABC, indices_C, 1,
            [&](stride_type next_A_AB, stride_type next_C_AC)
            {
                stride_type idx_B = idx_B0;

                for_each_match<true, false>(idx_B, next_B_ABC, indices_B, 1,
                                           idx_C,  next_C_AC, indices_C, 2,
                [&](stride_type next_B_AB)
                {
                    if (indices_C[idx_C].factor == T(0)) return;

                    for (unsigned irrep_AB0 = 0;irrep_AB0 < nirrep;irrep_AB0++)
                    {
                        unsigned irrep_AB = irrep_AB0;
                        unsigned irrep_AC = B.irrep()^C.irrep()^irrep_AB0;
                        unsigned irrep_BC = A.irrep()^C.irrep()^irrep_AB0;

                        for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
                        for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
                        for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                        if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
                        if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                        if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                        for (stride_type block_ABC = 0;block_ABC < group_ABC.dense_nblock;block_ABC++)
                        for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                        for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                        {
                            tasks.visit(idx++,
                            [&,idx_A,idx_B,idx_C,next_A_AB,next_B_AB,
                             irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC,block_ABC]
                            (const communicator& subcomm)
                            {
                                auto local_irreps_A = irreps_A;
                                auto local_irreps_B = irreps_B;
                                auto local_irreps_C = irreps_C;

                                assign_irreps(group_ABC.dense_ndim, irrep_ABC, nirrep, block_ABC,
                                              local_irreps_A, group_ABC.dense_idx[0],
                                              local_irreps_B, group_ABC.dense_idx[1],
                                              local_irreps_C, group_ABC.dense_idx[2]);

                                assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                              local_irreps_A, group_AC.dense_idx[0],
                                              local_irreps_C, group_AC.dense_idx[1]);

                                assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                              local_irreps_B, group_BC.dense_idx[0],
                                              local_irreps_C, group_BC.dense_idx[1]);

                                if (is_block_empty(dpd_C, local_irreps_C)) return;

                                auto local_C = dpd_C(local_irreps_C);

                                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                                {
                                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                                  local_irreps_A, group_AB.dense_idx[0],
                                                  local_irreps_B, group_AB.dense_idx[1]);

                                    if (is_block_empty(dpd_A, local_irreps_A)) continue;

                                    auto local_idx_A = idx_A;
                                    auto local_idx_B = idx_B;
                                    auto local_A = dpd_A(local_irreps_A);
                                    auto local_B = dpd_B(local_irreps_B);

                                    len_vector len_ABC;
                                    stride_vector stride_A_ABC, stride_B_ABC, stride_C_ABC;
                                    stride_type off_A_ABC, off_B_ABC, off_C_ABC;
                                    get_local_geometry(indices_A[local_idx_A].idx[0], group_ABC, len_ABC,
                                                       local_A, stride_A_ABC, 0,
                                                       local_B, stride_B_ABC, 1,
                                                       local_C, stride_C_ABC, 2);
                                    get_local_offset(indices_A[local_idx_A].idx[0], group_ABC,
                                                     local_A, off_A_ABC, 0,
                                                     local_B, off_B_ABC, 1,
                                                     local_C, off_C_ABC, 2);

                                    len_vector len_AC;
                                    stride_vector stride_A_AC, stride_C_AC;
                                    stride_type off_A_AC, off_C_AC;
                                    get_local_geometry(indices_A[local_idx_A].idx[1], group_AC, len_AC,
                                                       local_A, stride_A_AC, 0,
                                                       local_C, stride_C_AC, 1);
                                    get_local_offset(indices_A[local_idx_A].idx[1], group_AC,
                                                     local_A, off_A_AC, 0,
                                                     local_C, off_C_AC, 1);

                                    len_vector len_BC;
                                    stride_vector stride_B_BC, stride_C_BC;
                                    stride_type off_B_BC, off_C_BC;
                                    get_local_geometry(indices_B[local_idx_B].idx[1], group_BC, len_BC,
                                                       local_B, stride_B_BC, 0,
                                                       local_C, stride_C_BC, 1);
                                    get_local_offset(indices_B[local_idx_B].idx[1], group_BC,
                                                     local_B, off_B_BC, 0,
                                                     local_C, off_C_BC, 1);

                                    len_vector len_AB;
                                    stride_vector stride_A_AB, stride_B_AB;
                                    get_local_geometry(indices_A[local_idx_A].idx[2], group_AB, len_AB,
                                                       local_A, stride_A_AB, 0,
                                                       local_B, stride_B_AB, 1);

                                    auto data_C = local_C.data() + indices_C[idx_C].offset +
                                                  off_C_AC + off_C_BC + off_C_ABC;

                                    for_each_match<false, false>(local_idx_A, next_A_AB, indices_A, 2,
                                                                local_idx_B, next_B_AB, indices_B, 2,
                                    [&]
                                    {
                                        auto factor = alpha*indices_A[local_idx_A].factor*
                                                            indices_B[local_idx_B].factor*
                                                            indices_C[idx_C].factor;
                                        if (factor == T(0)) return;

                                        stride_type off_A_AB, off_B_AB;
                                        get_local_offset(indices_A[local_idx_A].idx[2], group_AB,
                                                         local_A, off_A_AB, 0,
                                                         local_B, off_B_AB, 1);

                                        auto data_A = local_A.data() + indices_A[local_idx_A].offset +
                                                      off_A_AB + off_A_AC + off_A_ABC;
                                        auto data_B = local_B.data() + indices_B[local_idx_B].offset +
                                                      off_B_AB + off_B_BC + off_B_ABC;

                                        mult(subcomm, cfg,
                                             len_AB, len_AC, len_BC, len_ABC,
                                             factor, conj_A, data_A, stride_A_AB, stride_A_AC, stride_A_ABC,
                                                     conj_B, data_B, stride_B_AB, stride_B_BC, stride_B_ABC,
                                               T(1),  false, data_C, stride_C_AC, stride_C_BC, stride_C_ABC);
                                    });
                                }
                            });
                        }
                    }
                });
            });
        });
    });
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                   bool conj_B, const indexed_dpd_varray_view<const T>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          T  beta, bool conj_C, const indexed_dpd_varray_view<      T>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    if (beta == T(0))
    {
        set(comm, cfg, T(0), C, range(C.dimension()));
    }
    else if (beta != T(1) || (is_complex<T>::value && conj_C))
    {
        scale(comm, cfg, beta, conj_C, C, range(C.dimension()));
    }

    for (unsigned i = 0;i < idx_A_AB.size();i++)
    {
        if (idx_A_AB[i] >= A.dense_dimension() &&
            idx_B_AB[i] >= B.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_AB[i] - A.dense_dimension()) !=
                B.indexed_irrep(idx_B_AB[i] - B.dense_dimension())) return;
        }
    }

    for (unsigned i = 0;i < idx_A_AC.size();i++)
    {
        if (idx_A_AC[i] >= A.dense_dimension() &&
            idx_C_AC[i] >= C.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_AC[i] - A.dense_dimension()) !=
                C.indexed_irrep(idx_C_AC[i] - C.dense_dimension())) return;
        }
    }

    for (unsigned i = 0;i < idx_B_BC.size();i++)
    {
        if (idx_B_BC[i] >= B.dense_dimension() &&
            idx_C_BC[i] >= C.dense_dimension())
        {
            if (B.indexed_irrep(idx_B_BC[i] - B.dense_dimension()) !=
                C.indexed_irrep(idx_C_BC[i] - C.dense_dimension())) return;
        }
    }

    for (unsigned i = 0;i < idx_A_ABC.size();i++)
    {
        if (idx_A_ABC[i] >= A.dense_dimension() &&
            idx_B_ABC[i] >= B.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_ABC[i] - A.dense_dimension()) !=
                B.indexed_irrep(idx_B_ABC[i] - B.dense_dimension())) return;
        }

        if (idx_A_ABC[i] >= A.dense_dimension() &&
            idx_C_ABC[i] >= C.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_ABC[i] - A.dense_dimension()) !=
                C.indexed_irrep(idx_C_ABC[i] - C.dense_dimension())) return;
        }

        if (idx_B_ABC[i] >= B.dense_dimension() &&
            idx_C_ABC[i] >= C.dense_dimension())
        {
            if (B.indexed_irrep(idx_B_ABC[i] - B.dense_dimension()) !=
                C.indexed_irrep(idx_C_ABC[i] - C.dense_dimension())) return;
        }
    }

    if (dpd_impl == FULL)
    {
        mult_full(comm, cfg,
                  alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                         conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                 C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
    else if (idx_C_ABC.empty())
    {
        contract_block_nofuse(comm, cfg,
                              alpha, conj_A, A, idx_A_AB, idx_A_AC,
                                     conj_B, B, idx_B_AB, idx_B_BC,
                                             C, idx_C_AC, idx_C_BC);
    }
    else
    {
        mult_block(comm, cfg,
                   alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                  C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A, \
                   const dim_vector& idx_A_AB, \
                   const dim_vector& idx_A_AC, \
                   const dim_vector& idx_A_ABC, \
                            bool conj_B, const indexed_dpd_varray_view<const T>& B, \
                   const dim_vector& idx_B_AB, \
                   const dim_vector& idx_B_BC, \
                   const dim_vector& idx_B_ABC, \
                   T  beta, bool conj_C, const indexed_dpd_varray_view<      T>& C, \
                   const dim_vector& idx_C_AC, \
                   const dim_vector& idx_C_BC, \
                   const dim_vector& idx_C_ABC);
#include "configs/foreach_type.h"

}
}
