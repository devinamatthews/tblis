#include "tblis.hpp"

extern blksz_t* gemm_mc;
extern blksz_t* gemm_nc;
extern blksz_t* gemm_kc;

namespace tblis
{

#ifdef BLIS_ENABLE_PTHREADS
static pthread_mutex_t tblis_initialize_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

void tblis_init()
{
#ifdef BLIS_ENABLE_OPENMP
    _Pragma( "omp critical (tblis_init)" )
#endif
#ifdef BLIS_ENABLE_PTHREADS
    pthread_mutex_lock( &tblis_initialize_mutex );
#endif

    {
        bli_init();

        dim_t block_size[3][BLIS_DT_HI+1];
        dim_t block_inc[3] = {2, 2, 1};
        blksz_t* block_sizes[3] = {gemm_mc, gemm_nc, gemm_kc};

        for (int dt = BLIS_DT_LO;dt <= BLIS_DT_HI;++dt)
        {
            for (int param = 0;param < 3;param++)
            {
                block_size[param][dt] = bli_blksz_get_max((num_t)dt, block_sizes[param]);
                bli_blksz_set_max(block_size[param][dt]+block_inc[param], (num_t)dt, block_sizes[param]);
            }
        }

        bli_mem_reinit();

        for (int dt = BLIS_DT_LO;dt <= BLIS_DT_HI;++dt)
        {
            for (int param = 0;param < 3;param++)
            {
                bli_blksz_set_max(block_size[param][dt], (num_t)dt, block_sizes[param]);
            }
        }
    }

#ifdef BLIS_ENABLE_PTHREADS
    pthread_mutex_unlock( &tblis_initialize_mutex );
#endif
}

void tblis_finalize()
{
    bli_finalize();
}

}
