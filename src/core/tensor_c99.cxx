#include "tensor.h"
#include "core/tensor_templates.hpp"

extern "C"
{

int tensor_smult(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                    float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_mult(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                      B, ndim_B, len_B, stride_B, idx_B,
                                beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_dmult(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_mult(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                      B, ndim_B, len_B, stride_B, idx_B,
                                beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_cmult(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                 scomplex  beta,       scomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_mult(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                      B, ndim_B, len_B, stride_B, idx_B,
                                beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_zmult(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                 dcomplex  beta,       dcomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_mult(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                      B, ndim_B, len_B, stride_B, idx_B,
                                beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_scontract(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                        float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_contract(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                          B, ndim_B, len_B, stride_B, idx_B,
                                    beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_dcontract(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_contract(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                          B, ndim_B, len_B, stride_B, idx_B,
                                    beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_ccontract(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     scomplex  beta,       scomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_contract(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                          B, ndim_B, len_B, stride_B, idx_B,
                                    beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_zcontract(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     dcomplex  beta,       dcomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_contract(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                          B, ndim_B, len_B, stride_B, idx_B,
                                    beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_sweight(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                      float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_weight(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                        B, ndim_B, len_B, stride_B, idx_B,
                                  beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_dweight(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_weight(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                        B, ndim_B, len_B, stride_B, idx_B,
                                  beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_cweight(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   scomplex  beta,       scomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_weight(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                        B, ndim_B, len_B, stride_B, idx_B,
                                  beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_zweight(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   dcomplex  beta,       dcomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_weight(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                        B, ndim_B, len_B, stride_B, idx_B,
                                  beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_souter_prod(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                          float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_outer_prod(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                            B, ndim_B, len_B, stride_B, idx_B,
                                      beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_douter_prod(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                         double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_outer_prod(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                            B, ndim_B, len_B, stride_B, idx_B,
                                      beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_couter_prod(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       scomplex  beta,       scomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_outer_prod(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                            B, ndim_B, len_B, stride_B, idx_B,
                                      beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_zouter_prod(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       dcomplex  beta,       dcomplex* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tensor::tensor_outer_prod(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                            B, ndim_B, len_B, stride_B, idx_B,
                                      beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_ssum(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                   float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_sum(alpha, A, ndim_A, len_A, stride_A, idx_A,
                               beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dsum(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_sum(alpha, A, ndim_A, len_A, stride_A, idx_A,
                               beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_csum(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                scomplex  beta,       scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_sum(alpha, A, ndim_A, len_A, stride_A, idx_A,
                               beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_zsum(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                dcomplex  beta,       dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_sum(alpha, A, ndim_A, len_A, stride_A, idx_A,
                               beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_strace(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                     float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_trace(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                 beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dtrace(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                    double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_trace(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                 beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_ctrace(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  scomplex  beta,       scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_trace(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                 beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_ztrace(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  dcomplex  beta,       dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_trace(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                 beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_sreplicate(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                         float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_replicate(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dreplicate(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                        double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_replicate(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_creplicate(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      scomplex  beta,       scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_replicate(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_zreplicate(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      dcomplex  beta,       dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_replicate(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_stranspose(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                         float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_transpose(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dtranspose(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                        double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_transpose(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_ctranspose(scomplex alpha, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      scomplex  beta,       scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_transpose(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_ztranspose(dcomplex alpha, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      dcomplex  beta,       dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tensor::tensor_transpose(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_sdot(const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,    float* val)
{
    return tensor::tensor_dot(A, ndim_A, len_A, stride_A, idx_A,
                              B, ndim_B, len_B, stride_B, idx_B, *val);
}

int tensor_ddot(const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,   double* val)
{
    return tensor::tensor_dot(A, ndim_A, len_A, stride_A, idx_A,
                              B, ndim_B, len_B, stride_B, idx_B, *val);
}

int tensor_cdot(const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const scomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B, scomplex* val)
{
    return tensor::tensor_dot(A, ndim_A, len_A, stride_A, idx_A,
                              B, ndim_B, len_B, stride_B, idx_B, *val);
}

int tensor_zdot(const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const dcomplex* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B, dcomplex* val)
{
    return tensor::tensor_dot(A, ndim_A, len_A, stride_A, idx_A,
                              B, ndim_B, len_B, stride_B, idx_B, *val);
}

int tensor_sscale(   float alpha,    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tensor::tensor_scale(alpha, A, ndim_A, len_A, stride_A, idx_A);
}

int tensor_dscale(  double alpha,   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tensor::tensor_scale(alpha, A, ndim_A, len_A, stride_A, idx_A);
}

int tensor_cscale(scomplex alpha, scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tensor::tensor_scale(alpha, A, ndim_A, len_A, stride_A, idx_A);
}

int tensor_zscale(dcomplex alpha, dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tensor::tensor_scale(alpha, A, ndim_A, len_A, stride_A, idx_A);
}

int tensor_sreduce(reduce_t op, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,    float* val, inc_t* idx)
{
    return tensor::tensor_reduce(op, A, ndim_A, len_A, stride_A, idx_A, *val, *idx);
}

int tensor_dreduce(reduce_t op, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,   double* val, inc_t* idx)
{
    return tensor::tensor_reduce(op, A, ndim_A, len_A, stride_A, idx_A, *val, *idx);
}

int tensor_creduce(reduce_t op, const scomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A, scomplex* val, inc_t* idx)
{
    return tensor::tensor_reduce(op, A, ndim_A, len_A, stride_A, idx_A, *val, *idx);
}

int tensor_zreduce(reduce_t op, const dcomplex* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A, dcomplex* val, inc_t* idx)
{
    return tensor::tensor_reduce(op, A, ndim_A, len_A, stride_A, idx_A, *val, *idx);
}

siz_t tensor_size(gint_t ndim, const dim_t* len, const inc_t* stride)
{
    return tensor::tensor_size(ndim, len, stride);
}

siz_t tensor_storage_size(gint_t ndim, const dim_t* len, const inc_t* stride)
{
    return tensor::tensor_storage_size(ndim, len, stride);
}

}
