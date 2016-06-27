#ifndef _TBLIS_HPP_
#define _TBLIS_HPP_

#include "tblis_config.hpp"

namespace tblis
{

void tblis_init();

void tblis_finalize();

}

#include "tblis_util.hpp"

#define _DEFINED_SCOMPLEX
#define _DEFINED_DCOMPLEX
extern "C"
{
#include "bli_system.h"
#include "bli_config.h"
#include "bli_config_macro_defs.h"
#include "bli_type_defs.h"
#include "bli_macro_defs.h"
}

#include "tblis_tensor.hpp"
#include "tensor_templates.hpp"
#include "tblis_memory_pool.hpp"
#include "tblis_yield.hpp"
#include "tblis_mutex.hpp"
#include "tblis_barrier.hpp"
#include "tblis_thread.hpp"

#include "tblis_config.hpp"
#include "tblis_scatter_matrix.hpp"
#include "tblis_block_scatter_matrix.hpp"
#include "tblis_tensor_matrix.hpp"

#include "1v/tblis_addv.hpp"
#include "1v/tblis_asumv.hpp"
#include "1v/tblis_axpbyv.hpp"
#include "1v/tblis_axpyv.hpp"
#include "1v/tblis_copyv.hpp"
#include "1v/tblis_dotv.hpp"
#include "1v/tblis_normfv.hpp"
#include "1v/tblis_scal2v.hpp"
#include "1v/tblis_scalv.hpp"
#include "1v/tblis_setv.hpp"
#include "1v/tblis_sumv.hpp"
#include "1v/tblis_xpbyv.hpp"
#include "1v/tblis_zerov.hpp"

#include "1m/tblis_normfm.hpp"

#include "3/tblis_partm.hpp"
#include "3/tblis_packm.hpp"
#include "3/tblis_gemm_ukr.hpp"
#include "3/tblis_gemm.hpp"

#endif
