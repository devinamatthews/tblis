#ifndef TBLIS_NODES_PACKM_HPP
#define TBLIS_NODES_PACKM_HPP 1

#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#include <tblis/internal/types.hpp>
#include <tblis/internal/memory_pool.hpp>

#include <tblis/matrix/abstract_matrix.hpp>

namespace tblis
{

template <int Mat, MemoryPool& Pool, typename Child>
struct pack
{
    Child child;

    void operator()(const communicator& comm, const config& cfg,
                    abstract_matrix& A, abstract_matrix& B, abstract_matrix& C)
    {
        if (Mat == matrix_constants::MAT_A)
        {
            auto P = A.pack(comm, cfg, Mat, Pool);
            comm.barrier();
            child(comm, cfg, P, B, C);
            comm.barrier();
        }
        else
        {
            auto P = B.pack(comm, cfg, Mat, Pool);
            comm.barrier();
            child(comm, cfg, A, P, C);
            comm.barrier();
        }
    }
};

template <MemoryPool& Pool, typename Child>
using pack_a = pack<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using pack_b = pack<matrix_constants::MAT_B, Pool, Child>;

}

#endif //TBLIS_NODES_PACKM_HPP
