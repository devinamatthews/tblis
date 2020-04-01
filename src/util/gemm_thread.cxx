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

    jc_nt = envtol("BLIS_JC_NT", jc_nt);
    ic_nt = envtol("BLIS_IC_NT", ic_nt);
    jr_nt = envtol("BLIS_JR_NT", jr_nt);
    ir_nt = envtol("BLIS_IR_NT", ir_nt);

    TBLIS_ASSERT(ir_nt*jr_nt*ic_nt*jc_nt == nthread);

    return {jc_nt, ic_nt, jr_nt, ir_nt};
}

}
