#ifndef _TBLIS_HPP_
#define _TBLIS_HPP_

#include "config.h"

namespace tblis
{

void tblis_init();

void tblis_finalize();

}

#include "util/util.hpp"

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

#include "core/tensor_iterator.hpp"
#include "core/tblis_tensor.hpp"
#include "core/tensor_iface.hpp"
#include "core/tensor_templates.hpp"
#include "core/tensor_slicer.hpp"
#include "core/tensor_partitioner.hpp"
#include "core/tblis_memory_pool.hpp"
#include "core/tblis_mutex.hpp"
#include "core/tblis_thread.hpp"

#include "blis-like/tblis_config.hpp"
#include "blis-like/tblis_scatter_matrix.hpp"
#include "blis-like/tblis_block_scatter_matrix.hpp"
#include "blis-like/tblis_tensor_matrix.hpp"

#include "blis-like/1v/tblis_addv.hpp"
#include "blis-like/1v/tblis_asumv.hpp"
#include "blis-like/1v/tblis_axpbyv.hpp"
#include "blis-like/1v/tblis_axpyv.hpp"
#include "blis-like/1v/tblis_copyv.hpp"
#include "blis-like/1v/tblis_dotv.hpp"
#include "blis-like/1v/tblis_normfv.hpp"
#include "blis-like/1v/tblis_scal2v.hpp"
#include "blis-like/1v/tblis_scalv.hpp"
#include "blis-like/1v/tblis_setv.hpp"
#include "blis-like/1v/tblis_sumv.hpp"
#include "blis-like/1v/tblis_xpbyv.hpp"
#include "blis-like/1v/tblis_zerov.hpp"

#include "blis-like/1m/tblis_normfm.hpp"

#include "blis-like/3/tblis_partm.hpp"
#include "blis-like/3/tblis_packm.hpp"
#include "blis-like/3/tblis_gemm_ukr.hpp"
#include "blis-like/3/tblis_gemm.hpp"

#endif
