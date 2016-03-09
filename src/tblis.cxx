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

        dim_t mc_s = bli_blksz_get_max(   BLIS_FLOAT, gemm_mc);
        dim_t mc_d = bli_blksz_get_max(  BLIS_DOUBLE, gemm_mc);
        dim_t mc_c = bli_blksz_get_max(BLIS_SCOMPLEX, gemm_mc);
        dim_t mc_z = bli_blksz_get_max(BLIS_DCOMPLEX, gemm_mc);
        dim_t nc_s = bli_blksz_get_max(   BLIS_FLOAT, gemm_nc);
        dim_t nc_d = bli_blksz_get_max(  BLIS_DOUBLE, gemm_nc);
        dim_t nc_c = bli_blksz_get_max(BLIS_SCOMPLEX, gemm_nc);
        dim_t nc_z = bli_blksz_get_max(BLIS_DCOMPLEX, gemm_nc);
        dim_t kc_s = bli_blksz_get_max(   BLIS_FLOAT, gemm_kc);
        dim_t kc_d = bli_blksz_get_max(  BLIS_DOUBLE, gemm_kc);
        dim_t kc_c = bli_blksz_get_max(BLIS_SCOMPLEX, gemm_kc);
        dim_t kc_z = bli_blksz_get_max(BLIS_DCOMPLEX, gemm_kc);

        bli_blksz_set_max(mc_s+2,    BLIS_FLOAT, gemm_mc);
        bli_blksz_set_max(mc_d+2,   BLIS_DOUBLE, gemm_mc);
        bli_blksz_set_max(mc_c+2, BLIS_SCOMPLEX, gemm_mc);
        bli_blksz_set_max(mc_z+2, BLIS_DCOMPLEX, gemm_mc);
        bli_blksz_set_max(nc_s+2,    BLIS_FLOAT, gemm_nc);
        bli_blksz_set_max(nc_d+2,   BLIS_DOUBLE, gemm_nc);
        bli_blksz_set_max(nc_c+2, BLIS_SCOMPLEX, gemm_nc);
        bli_blksz_set_max(nc_z+2, BLIS_DCOMPLEX, gemm_nc);
        bli_blksz_set_max(kc_s+1,    BLIS_FLOAT, gemm_kc);
        bli_blksz_set_max(kc_d+1,   BLIS_DOUBLE, gemm_kc);
        bli_blksz_set_max(kc_c+1, BLIS_SCOMPLEX, gemm_kc);
        bli_blksz_set_max(kc_z+1, BLIS_DCOMPLEX, gemm_kc);

        bli_mem_reinit();

        bli_blksz_set_max(mc_s,    BLIS_FLOAT, gemm_mc);
        bli_blksz_set_max(mc_d,   BLIS_DOUBLE, gemm_mc);
        bli_blksz_set_max(mc_c, BLIS_SCOMPLEX, gemm_mc);
        bli_blksz_set_max(mc_z, BLIS_DCOMPLEX, gemm_mc);
        bli_blksz_set_max(nc_s,    BLIS_FLOAT, gemm_nc);
        bli_blksz_set_max(nc_d,   BLIS_DOUBLE, gemm_nc);
        bli_blksz_set_max(nc_c, BLIS_SCOMPLEX, gemm_nc);
        bli_blksz_set_max(nc_z, BLIS_DCOMPLEX, gemm_nc);
        bli_blksz_set_max(kc_s,    BLIS_FLOAT, gemm_kc);
        bli_blksz_set_max(kc_d,   BLIS_DOUBLE, gemm_kc);
        bli_blksz_set_max(kc_c, BLIS_SCOMPLEX, gemm_kc);
        bli_blksz_set_max(kc_z, BLIS_DCOMPLEX, gemm_kc);
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
