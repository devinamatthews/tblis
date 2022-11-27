#ifndef TBLIS_GEMM_THREAD_HPP
#define TBLIS_GEMM_THREAD_HPP

#include <tblis/internal/configs.hpp>
#include <tblis/internal/types.hpp>

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

gemm_thread_config make_gemm_thread_config(type_t type, const config& cfg,
    int nthread, len_type m, len_type n, len_type);

}

#endif
