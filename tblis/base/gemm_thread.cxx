#include "gemm_thread.hpp"
#include "env.hpp"

namespace tblis
{

gemm_thread_config make_gemm_thread_config(type_t type, const config& cfg,
    int nthread, len_type m, len_type n, len_type)
{
    int ic_nt, jc_nt, ir_nt, jr_nt;

    std::tie(ic_nt, jc_nt) =
        partition_2x2(nthread, m*cfg.m_thread_ratio.value(type),
                               n*cfg.n_thread_ratio.value(type));

    for (ir_nt = cfg.mr_max_thread.value(type);ir_nt > 1;ir_nt--)
    {
        if (ic_nt%ir_nt == 0)
        {
            ic_nt /= ir_nt;
            break;
        }
    }

    for (jr_nt = cfg.nr_max_thread.value(type);jr_nt > 1;jr_nt--)
    {
        if (jc_nt%jr_nt == 0)
        {
            jc_nt /= jr_nt;
            break;
        }
    }

    static auto jc_nt_env = envtol("BLIS_JC_NT", -1);
    static auto ic_nt_env = envtol("BLIS_IC_NT", -1);
    static auto jr_nt_env = envtol("BLIS_JR_NT", -1);
    static auto ir_nt_env = envtol("BLIS_IR_NT", -1);

    jc_nt = jc_nt_env != -1 ? jc_nt_env : jc_nt;
    ic_nt = ic_nt_env != -1 ? ic_nt_env : ic_nt;
    jr_nt = jr_nt_env != -1 ? jr_nt_env : jr_nt;
    ir_nt = ir_nt_env != -1 ? ir_nt_env : ir_nt;

    TBLIS_ASSERT(ir_nt*jr_nt*ic_nt*jc_nt == nthread);

    return {jc_nt, ic_nt, jr_nt, ir_nt};
}

}
