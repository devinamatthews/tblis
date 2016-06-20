#include "tblis.hpp"

using namespace blis;

static const scomplex* conv(const complex float* x) { return reinterpret_cast<const scomplex*>(x); }
static       scomplex* conv(      complex float* x) { return reinterpret_cast<      scomplex*>(x); }
static const scomplex& conv(const complex float& x) { return reinterpret_cast<const scomplex&>(x); }
static       scomplex& conv(      complex float& x) { return reinterpret_cast<      scomplex&>(x); }
static const dcomplex* conv(const complex double* x) { return reinterpret_cast<const dcomplex*>(x); }
static       dcomplex* conv(      complex double* x) { return reinterpret_cast<      dcomplex*>(x); }
static const dcomplex& conv(const complex double& x) { return reinterpret_cast<const dcomplex&>(x); }
static       dcomplex& conv(      complex double& x) { return reinterpret_cast<      dcomplex&>(x); }

extern "C"
{

int tensor_smult(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                    float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_mult(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                      B, ndim_B, len_B, stride_B, idx_B,
                                beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_dmult(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_mult(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                      B, ndim_B, len_B, stride_B, idx_B,
                                beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_cmult(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                 complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_mult(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                            conv(B), ndim_B, len_B, stride_B, idx_B,
                               conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_zmult(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                 complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_mult(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                            conv(B), ndim_B, len_B, stride_B, idx_B,
                               conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_scontract(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                        float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_contract(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                          B, ndim_B, len_B, stride_B, idx_B,
                                    beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_dcontract(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_contract(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                          B, ndim_B, len_B, stride_B, idx_B,
                                    beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_ccontract(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_contract(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                                conv(B), ndim_B, len_B, stride_B, idx_B,
                                   conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_zcontract(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_contract(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                                conv(B), ndim_B, len_B, stride_B, idx_B,
                                   conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_sweight(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                      float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_weight(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                        B, ndim_B, len_B, stride_B, idx_B,
                                  beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_dweight(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_weight(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                        B, ndim_B, len_B, stride_B, idx_B,
                                  beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_cweight(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_weight(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                              conv(B), ndim_B, len_B, stride_B, idx_B,
                                 conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_zweight(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_weight(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                              conv(B), ndim_B, len_B, stride_B, idx_B,
                                 conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_souter_prod(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                          float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_outer_prod(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                            B, ndim_B, len_B, stride_B, idx_B,
                                      beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_douter_prod(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                         double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_outer_prod(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                            B, ndim_B, len_B, stride_B, idx_B,
                                      beta, C, ndim_C, len_C, stride_C, idx_C);
}

int tensor_couter_prod(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_outer_prod(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                                  conv(B), ndim_B, len_B, stride_B, idx_B,
                                     conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_zouter_prod(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C)
{
    return tblis::tensor_outer_prod(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                                  conv(B), ndim_B, len_B, stride_B, idx_B,
                                     conv( beta), conv(C), ndim_C, len_C, stride_C, idx_C);
}

int tensor_ssum(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                   float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_sum(alpha, A, ndim_A, len_A, stride_A, idx_A,
                               beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dsum(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_sum(alpha, A, ndim_A, len_A, stride_A, idx_A,
                               beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_csum(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_sum(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                              conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_zsum(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_sum(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                              conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_strace(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                     float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_trace(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                 beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dtrace(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                    double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_trace(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                 beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_ctrace(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_trace(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_ztrace(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_trace(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_sreplicate(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                         float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_replicate(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dreplicate(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                        double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_replicate(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_creplicate(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_replicate(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                    conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_zreplicate(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_replicate(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                    conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_stranspose(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                         float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_transpose(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_dtranspose(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                        double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_transpose(alpha, A, ndim_A, len_A, stride_A, idx_A,
                                     beta, B, ndim_B, len_B, stride_B, idx_B);
}

int tensor_ctranspose(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_transpose(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                    conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_ztranspose(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B)
{
    return tblis::tensor_transpose(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A,
                                    conv( beta), conv(B), ndim_B, len_B, stride_B, idx_B);
}

int tensor_sdot(const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,    float* val)
{
    return tblis::tensor_dot(A, ndim_A, len_A, stride_A, idx_A,
                              B, ndim_B, len_B, stride_B, idx_B, *val);
}

int tensor_ddot(const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,   double* val)
{
    return tblis::tensor_dot(A, ndim_A, len_A, stride_A, idx_A,
                              B, ndim_B, len_B, stride_B, idx_B, *val);
}

int tensor_cdot(const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B, complex float* val)
{
    return tblis::tensor_dot(conv(A), ndim_A, len_A, stride_A, idx_A,
                              conv(B), ndim_B, len_B, stride_B, idx_B, conv(*val));
}

int tensor_zdot(const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B, complex double* val)
{
    return tblis::tensor_dot(conv(A), ndim_A, len_A, stride_A, idx_A,
                              conv(B), ndim_B, len_B, stride_B, idx_B, conv(*val));
}

int tensor_sscale(   float alpha,    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tblis::tensor_scale(alpha, A, ndim_A, len_A, stride_A, idx_A);
}

int tensor_dscale(  double alpha,   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tblis::tensor_scale(alpha, A, ndim_A, len_A, stride_A, idx_A);
}

int tensor_cscale(complex float alpha, complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tblis::tensor_scale(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A);
}

int tensor_zscale(complex double alpha, complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A)
{
    return tblis::tensor_scale(conv(alpha), conv(A), ndim_A, len_A, stride_A, idx_A);
}

int tensor_sreduce(reduce_t op, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,    float* val, inc_t* idx)
{
    return tblis::tensor_reduce(op, A, ndim_A, len_A, stride_A, idx_A, *val, *idx);
}

int tensor_dreduce(reduce_t op, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,   double* val, inc_t* idx)
{
    return tblis::tensor_reduce(op, A, ndim_A, len_A, stride_A, idx_A, *val, *idx);
}

int tensor_creduce(reduce_t op, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A, complex float* val, inc_t* idx)
{
    return tblis::tensor_reduce(op, conv(A), ndim_A, len_A, stride_A, idx_A, conv(*val), *idx);
}

int tensor_zreduce(reduce_t op, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A, complex double* val, inc_t* idx)
{
    return tblis::tensor_reduce(op, conv(A), ndim_A, len_A, stride_A, idx_A, conv(*val), *idx);
}

siz_t tensor_size(gint_t ndim, const dim_t* len, const inc_t* stride)
{
    return tblis::tensor_size(ndim, len, stride);
}

siz_t tensor_storage_size(gint_t ndim, const dim_t* len, const inc_t* stride)
{
    return tblis::tensor_storage_size(ndim, len, stride);
}

}
