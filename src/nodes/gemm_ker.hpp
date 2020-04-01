#ifndef _TBLIS_NODES_GEMM_MKR_HPP_
#define _TBLIS_NODES_GEMM_MKR_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "configs/configs.hpp"

#include "matrix/packed_matrix.hpp"

namespace tblis
{

extern MemoryPool BuffersForC;

template <MemoryPool& Pool=BuffersForC>
struct gemm_kernel
{
    void operator()(const communicator& comm, const config& cfg,
                    abstract_matrix& A, abstract_matrix& B, abstract_matrix& C) const
    {
        C.gemm(comm, cfg, Pool, A, B);
    }
};

}

#endif
