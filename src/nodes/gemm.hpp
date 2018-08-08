#ifndef _TBLIS_NODES_GEMM_HPP_
#define _TBLIS_NODES_GEMM_HPP_

#include "partm.hpp"
#include "packm.hpp"
#include "matrify.hpp"
#include "gemm_mkr.hpp"
#include "gemm_ukr.hpp"

namespace tblis
{

extern MemoryPool BuffersForA, BuffersForB, BuffersForScatter;

template <template <typename> class NodeType, typename T>
struct node_depth
{
    static constexpr unsigned value = node_depth<NodeType,decltype(std::declval<T>().child)>::value+1;
};

template <typename T>
struct node_depth<partition_gemm_mc, partition_gemm_mc<T>>
{
    static constexpr unsigned value = 0;
};

template <typename T>
struct node_depth<partition_gemm_nc, partition_gemm_nc<T>>
{
    static constexpr unsigned value = 0;
};

template <typename T>
struct node_depth<partition_gemm_kc, partition_gemm_kc<T>>
{
    static constexpr unsigned value = 0;
};

template <typename T>
struct node_depth<partition_gemm_mr, partition_gemm_mr<T>>
{
    static constexpr unsigned value = 0;
};

template <typename T>
struct node_depth<partition_gemm_nr, partition_gemm_nr<T>>
{
    static constexpr unsigned value = 0;
};

template <template <typename> class NodeType>
struct node_depth<NodeType, gemm_micro_kernel>
{
    static constexpr unsigned value = 0;
};

template <unsigned Depth, typename T>
struct node_type
{
    typedef typename node_type<Depth-1, decltype(std::declval<T>().child)>::type type;
};

template <typename T>
struct node_type<0, T>
{
    typedef T type;
};

template <unsigned Depth>
struct node_at_depth
{
    template <typename T>
    typename node_type<Depth,T>::type& operator()(T& tree) const
    {
        return node_at_depth<Depth-1>{}(tree.child);
    }
};

template <>
struct node_at_depth<0>
{
    template <typename T>
    T& operator()(T& tree) const
    {
        return tree;
    }
};

template <template <typename> class NodeType, typename T>
typename node_type<node_depth<NodeType,T>::value,T>::type& node(T& tree)
{
    return node_at_depth<node_depth<NodeType,T>::value>{}(tree);
}

template <typename Child>
struct gemm
{
    Child child;

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, const MatrixA& A, const MatrixB& B, T beta, const MatrixC& C)
    {
        using namespace matrix_constants;

        const bool row_major = cfg.gemm_row_major.value<T>();
        const bool trans = C.stride(!row_major) == 1;

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);

        if (trans)
        {
            /*
             * Compute C^T = B^T * A^T instead
             */
            std::swap(m, n);
        }

        if (comm.master()) flops += 2*m*n*k;

        int nt = comm.num_threads();
        auto tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);

        communicator comm_nc =    comm.gang(TCI_EVENLY, tc.jc_nt);
        communicator comm_kc = comm_nc.gang(TCI_EVENLY,        1);
        communicator comm_mc = comm_kc.gang(TCI_EVENLY, tc.ic_nt);
        communicator comm_nr = comm_mc.gang(TCI_EVENLY, tc.jr_nt);
        communicator comm_mr = comm_nr.gang(TCI_EVENLY, tc.ir_nt);

        node<partition_gemm_nc>(child).subcomm = &comm_nc;
        node<partition_gemm_kc>(child).subcomm = &comm_kc;
        node<partition_gemm_mc>(child).subcomm = &comm_mc;
        node<partition_gemm_nr>(child).subcomm = &comm_nr;
        node<partition_gemm_mr>(child).subcomm = &comm_mr;

        if (trans)
        {
            /*
             * Compute C^T = B^T * A^T instead
             */
            auto At = A;
            auto Bt = B;
            auto Ct = C;

            At.transpose();
            Bt.transpose();
            Ct.transpose();

            child(comm, cfg, alpha, Bt, At, beta, Ct);
        }
        else
        {
            child(comm, cfg, alpha, A, B, beta, C);
        }
    }
};

using GotoGEMM = gemm<
                   partition_gemm_nc<
                     partition_gemm_kc<
                       pack_b<BuffersForB,
                         partition_gemm_mc<
                           pack_a<BuffersForA,
                             partition_gemm_nr<
                               partition_gemm_mr<
                                 gemm_micro_kernel>>>>>>>>;

using GotoGEMM2 = gemm<
                    partition_gemm_nc<
                      partition_gemm_kc<
                        pack_b<BuffersForB,
                          partition_gemm_mc<
                            pack_a<BuffersForA,
                              gemm_macro_kernel>>>>>>;

using TensorGEMM = gemm<
                     partition_gemm_nc<
                       partition_gemm_kc<
                         matrify_and_pack_b<BuffersForB,
                           partition_gemm_mc<
                             matrify_and_pack_a<BuffersForA,
                               matrify_c<BuffersForScatter,
                                 partition_gemm_nr<
                                   partition_gemm_mr<
                                     gemm_micro_kernel>>>>>>>>>;

}

#endif
