#ifndef _TBLIS_GEMM_THREAD_HPP_
#define _TBLIS_GEMM_THREAD_HPP_

#include "basic_types.h"

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

gemm_thread_config make_gemm_thread_config(
    int nthread, len_type m, len_type n, len_type k);

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
