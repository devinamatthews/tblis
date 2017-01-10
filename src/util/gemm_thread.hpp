#ifndef _TBLIS_GEMM_THREAD_HPP_
#define _TBLIS_GEMM_THREAD_HPP_

#include "basic_types.h"
#include "env.hpp"

namespace tblis
{

struct gemm_thread_config
{
    gemm_thread_config(int jc_nt_, int ic_nt_, int jr_nt_, int ir_nt_)
    : jc_nt(jc_nt_), ic_nt(ic_nt_), jr_nt(jr_nt_), ir_nt(ir_nt_) {}

    int jc_nt = 1;
    int ic_nt = 1;
    int jr_nt = 1;
    int ir_nt = 1;
};

template <typename T>
gemm_thread_config make_gemm_thread_config(const config& cfg,
    int nthread, len_type m, len_type n, len_type k)
{
    int ic_nt, jc_nt, ir_nt, jr_nt;

    std::tie(ic_nt, jc_nt) =
        partition_2x2(nthread, m*cfg.m_thread_ratio.value<T>(),
                               n*cfg.n_thread_ratio.value<T>());

    for (ir_nt = cfg.mr_max_thread.value<T>();ir_nt > 1;ir_nt--)
    {
        if (ic_nt%ir_nt == 0)
        {
            ic_nt /= ir_nt;
            break;
        }
    }

    for (jr_nt = cfg.nr_max_thread.value<T>();jr_nt > 1;jr_nt--)
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
