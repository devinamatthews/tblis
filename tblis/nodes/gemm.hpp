#ifndef TBLIS_NODES_GEMM_HPP
#define TBLIS_NODES_GEMM_HPP

#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#include <tblis/internal/types.hpp>
#include <tblis/internal/thread.hpp>
#include <tblis/internal/gemm_thread.hpp>

#include <tblis/nodes/gemm_ker.hpp>
#include <tblis/nodes/partm.hpp>
#include <tblis/nodes/packm.hpp>

namespace tblis
{

extern MemoryPool BuffersForA, BuffersForB, BuffersForC;

template <template <typename> class NodeType, typename T>
struct node_depth
{
    static constexpr int value = node_depth<NodeType,decltype(std::declval<T>().child)>::value+1;
};

template <typename T>
struct node_depth<partition_gemm_mc, partition_gemm_mc<T>>
{
    static constexpr int value = 0;
};

template <typename T>
struct node_depth<partition_gemm_nc, partition_gemm_nc<T>>
{
    static constexpr int value = 0;
};

template <typename T>
struct node_depth<partition_gemm_kc, partition_gemm_kc<T>>
{
    static constexpr int value = 0;
};

template <typename T>
struct node_depth<partition_gemm_mr, partition_gemm_mr<T>>
{
    static constexpr int value = 0;
};

template <typename T>
struct node_depth<partition_gemm_nr, partition_gemm_nr<T>>
{
    static constexpr int value = 0;
};

template <template <typename> class NodeType, MemoryPool& Pool>
struct node_depth<NodeType, gemm_kernel<Pool>>
{
    static constexpr int value = 0;
};

template <int Depth, typename T>
struct node_type
{
    typedef typename node_type<Depth-1, decltype(std::declval<T>().child)>::type type;
};

template <typename T>
struct node_type<0, T>
{
    typedef T type;
};

template <int Depth>
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

    void operator()(const communicator& comm, const config& cfg,
                    abstract_matrix A, abstract_matrix B,
                    abstract_matrix C)
    {
        using namespace matrix_constants;

        const bool row_major = cfg.gemm_row_major.value(C.type());
        const bool trans = C.row_major() != row_major;

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
        auto tc = make_gemm_thread_config(C.type(), cfg, nt, m, n, k);

        communicator comm_nc =    comm.gang(TCI_EVENLY, tc.jc_nt);
        communicator comm_kc = comm_nc.gang(TCI_EVENLY,        1);
        communicator comm_mc = comm_kc.gang(TCI_EVENLY, tc.ic_nt);

        node<partition_gemm_nc>(child).subcomm = &comm_nc;
        node<partition_gemm_kc>(child).subcomm = &comm_kc;
        node<partition_gemm_mc>(child).subcomm = &comm_mc;

        if (trans)
        {
            /*
             * Compute C^T = B^T * A^T instead
             */
            A.transpose();
            B.transpose();
            C.transpose();

            child(comm, cfg, B, A, C);
        }
        else
        {
            child(comm, cfg, A, B, C);
        }

        comm.barrier();
    }
};

using GotoGEMM = gemm<
                   partition_gemm_nc<
                     partition_gemm_kc<
                       pack_b<BuffersForB,
                         partition_gemm_mc<
                           pack_a<BuffersForA,
                             gemm_kernel<BuffersForC>
                           >
                         >
                       >
                     >
                   >
                 >;

}

#endif
