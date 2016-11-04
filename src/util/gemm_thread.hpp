#ifndef _TBLIS_GEMM_THREAD_HPP_
#define _TBLIS_GEMM_THREAD_HPP_

#include "basic_types.h"
#include "assert.h"
#include "thread.h"

#include "util/env.hpp"

namespace tblis
{

struct gemm_thread_config
{
    gemm_thread_config(int jc_nt, int ic_nt, int jr_nt, int ir_nt)
    : jc_nt(jc_nt), ic_nt(ic_nt), jr_nt(jr_nt), ir_nt(ir_nt) {}

    int jc_nt = 1;
    int ic_nt = 1;
    int jr_nt = 1;
    int ir_nt = 1;
};

gemm_thread_config make_gemm_thread_config(int nthread, len_type m, len_type n, len_type k)
{
    int ic_nt, jc_nt;
    std::tie(ic_nt, jc_nt) = partition_2x2(nthread, m*2, n);

    int ir_nt = 1, jr_nt;
    for (jr_nt = 3;jr_nt > 1;jr_nt--)
    {
        if (jc_nt%jr_nt == 0)
        {
            jc_nt /= jr_nt;
            break;
        }
    }

    jc_nt = envtol("BLIS_JC_NT", jc_nt);
    ic_nt = envtol("BLIS_IC_NT", ic_nt);
    jr_nt = envtol("BLIS_JR_NT", jr_nt);
    ir_nt = envtol("BLIS_IR_NT", ir_nt);

    TBLIS_ASSERT(ir_nt*jr_nt*ic_nt*jc_nt == nthread);

    return {jc_nt, ic_nt, jr_nt, ir_nt};
}

template <int N> struct step_helper;

template <>
struct step_helper<0>
{
    template <typename T>
    T& operator()(T& tree) const { return tree; }
};

template <int N>
struct step_helper
{
    template <typename T>
    auto operator()(T& tree) const -> decltype(step_helper<N-1>()(tree.child))
    {
        return step_helper<N-1>()(tree.child);
    }
};

template <int N, typename T>
auto step(T& tree) -> decltype(step_helper<N>()(tree))
{
    return step_helper<N>()(tree);
}

}

#endif
