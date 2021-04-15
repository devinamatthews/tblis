#ifndef TBLIS_BASE_KERNELS_H
#define TBLIS_BASE_KERNELS_H 1

#include <tblis/blis.h>
#include <tblis/base/types.h>

TBLIS_BEGIN_NAMESPACE

typedef
void (*addf_sum_ukr_t)(len_type m, len_type n,
         const void* alpha, bool conj_A, const void** A, stride_type inc_A,
                            bool conj_B, const void*  B, stride_type inc_B,
         const void*  beta, bool conj_C,       void*  C, stride_type inc_C);

typedef
void (*addf_rep_ukr_t)(len_type m, len_type n,
         const void* alpha, bool conj_A, const void*  A, stride_type inc_A,
                            bool conj_B, const void*  B, stride_type inc_B,
         const void*  beta, bool conj_C,       void** C, stride_type inc_C);
typedef
void (*dotf_ukr_t)(len_type m, len_type n,
         const void* alpha, bool conj_A, const void* A, stride_type rs_A, stride_type cs_A,
                            bool conj_B, const void* B, stride_type inc_B,
         const void*  beta, bool conj_C,       void* C, stride_type inc_C);

typedef
void (*trans_ukr_t)(len_type m, len_type n,
         const void* alpha, bool conj_A, const void* A, stride_type rs_A, stride_type cs_A,
         const void*  beta, bool conj_B,       void* B, stride_type rs_B, stride_type cs_B);

typedef
void (*add_ukr_t)(len_type n,
         const void* alpha, bool conj_A, const void* A, stride_type inc_A,
         const void*  beta, bool conj_B,       void* B, stride_type inc_B);

typedef
void (*dot_ukr_t)(len_type n,
         bool conj_A, const void* A, stride_type inc_A,
         bool conj_B, const void* B, stride_type inc_B, void* value);

typedef
void (*mult_ukr_t)(len_type n,
         const void* alpha, bool conj_A, const void* A, stride_type inc_A,
                            bool conj_B, const void* B, stride_type inc_B,
         const void*  beta, bool conj_C,       void* C, stride_type inc_C);

typedef
void (*reduce_ukr_t)(reduce_t op, len_type n,
         const void* A, stride_type inc_A, void* value, len_type& idx);

typedef
void (*scale_ukr_t)(len_type n,
         const void* alpha, bool conj_A, void* A, stride_type inc_A);

typedef
void (*set_ukr_t)(len_type n,
         const void* alpha, void* A, stride_type inc_A);

typedef
void (*shift_ukr_t)(len_type n,
         const void* alpha, const void* beta, bool conj_A, void* A, stride_type inc_A);

typedef
void (*update_nn_ukr_t)(stride_type m, stride_type n,
         const void* ab,
         const void* d, stride_type inc_d,
         const void* e, stride_type inc_e,
         const void* beta,
         void* c, stride_type rs_c, stride_type cs_c);

typedef
void (*update_ss_ukr_t)(stride_type m, stride_type n,
         const void* ab,
         const void* beta,
         void* c, const stride_type* rscat_c, const stride_type* cscat_c);

typedef
void (*gemm_ukr_t)(stride_type m, stride_type n, stride_type k,
         const void* a, const void* b,
         const void* beta,
         void* c, stride_type rs_c, stride_type cs_c,
         auxinfo_t* aux);

typedef
void (*pack_nn_ukr_t)(len_type m, len_type k, const void* alpha, bool conj,
         const void* p_a, stride_type rs_a, stride_type cs_a,
         const void* p_d, stride_type inc_d,
         const void* p_e, stride_type inc_e,
         void* p_ap);

typedef
void (*pack_ss_ukr_t)(len_type m, len_type k, const void* alpha, bool conj,
         const void* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         void* p_ap);

typedef
void (*pack_nb_ukr_t)(len_type m, len_type k, const void* alpha, bool conj,
         const void* p_a, stride_type rs_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         void* p_ap);

typedef
void (*pack_ss_scal_ukr_t)(len_type m, len_type k, const void* alpha, bool conj,
         const void* p_a, const stride_type* rscat_a, const void* rscale_a,
         const stride_type* cscat_a, const void* cscale_a, void* p_ap);

TBLIS_END_NAMESPACE

#endif //TBLIS_BASE_KERNELS_H
