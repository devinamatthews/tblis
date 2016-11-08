#include "gemm_thread.hpp"

#include "assert.h"
#include "thread.h"
#include "env.hpp"

namespace tblis
{

gemm_thread_config make_gemm_thread_config(
    int nthread, len_type m, len_type n, len_type k)
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

}
