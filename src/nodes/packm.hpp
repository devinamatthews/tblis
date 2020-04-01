#ifndef _TBLIS_NODES_PACKM_HPP_
#define _TBLIS_NODES_PACKM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "memory/memory_pool.hpp"
#include "matrix/abstract_matrix.hpp"
#include "configs/configs.hpp"

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
            child(comm, cfg, P, B, C);
        }
        else
        {
            auto P = B.pack(comm, cfg, Mat, Pool);
            child(comm, cfg, A, P, C);
        }
    }
};

template <MemoryPool& Pool, typename Child>
using pack_a = pack<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using pack_b = pack<matrix_constants::MAT_B, Pool, Child>;

}

#endif
