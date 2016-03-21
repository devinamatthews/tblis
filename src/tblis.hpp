#ifndef _TBLIS_HPP_
#define _TBLIS_HPP_

#include "config.h"

#define BLIS_DISABLE_BLAS2BLIS
#include "blis++/blis++.hpp"

#include "tensor.h"

namespace tblis
{

using namespace blis;

void tblis_init();

void tblis_finalize();

}

#include "core/tensor_iterator.hpp"
#include "core/tensor_class.hpp"
#include "core/tensor_iface.hpp"
#include "core/tensor_templates.hpp"
#include "core/tensor_slicer.hpp"
#include "core/tensor_partitioner.hpp"

#include "blis-like/tblis_constants.hpp"
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
