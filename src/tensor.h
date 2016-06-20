#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <complex.h>

enum reduce_t
{
    REDUCE_SUM      = 0,
    REDUCE_SUM_ABS  = 1,
    REDUCE_MAX      = 2,
    REDUCE_MAX_ABS  = 3,
    REDUCE_MIN      = 4,
    REDUCE_MIN_ABS  = 5,
    REDUCE_NORM_1   = REDUCE_SUM_ABS,
    REDUCE_NORM_2   = 6,
    REDUCE_NORM_INF = REDUCE_MAX_ABS
};

/**
 * Main interface definitions
 *
 * In all cases, a repeated index label, either within the same tensor or in more than one tensor requires that the edge length be the same
 * in all labeled indices. 0-dimension tensors (scalars) are
 * allowed in all functions. In this case, the arguments len_*, stride_*, and idx_* may be NULL and will not be referenced. The data array for a scalar
 * must NOT be NULL, and should be of size 1. The special case of beta == +/-0.0 will overwrite special floating point values such as NaN and Inf
 * in the output.
 */

/**
 * Multiply two tensors together and sum onto a third
 *
 * This form generalizes contraction and weighting with the unary operations trace, transpose, and replicate. Note that
 * the binary contraction operation is similar in form to the unary trace operation, while the binary weighting operation is similar in form to the
 * unary diagonal operation. Any combination of these operations may be performed.
 */
int tensor_smult(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                    float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_dmult(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_cmult(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                 complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_zmult(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                 const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                 complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

/**
 * Contract two tensors into a third
 *
 * The general form for a contraction is ab...ef... * ef...cd... -> ab...cd... where the indices ef... will be summed over.
 * Indices may be transposed in any tensor. Any index group may be empty (in the case that ef... is empty, this reduces to an outer product).
 */
int tensor_scontract(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                        float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_dcontract(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_ccontract(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_zcontract(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                     const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

/**
 * Weight a tensor by a second and sum onto a third
 *
 * The general form for a weighting is ab...ef... * ef...cd... -> ab...cd...ef... with no indices being summed over.
 * Indices may be transposed in any tensor. Any index group may be empty
 * (in the case that ef... is empty, this reduces to an outer product).
 */
int tensor_sweight(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                      float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_dweight(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                     double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_cweight(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_zweight(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                   const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                   complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

/**
 * Sum the outer product of two tensors onto a third
 *
 * The general form for an outer product is ab... * cd... -> ab...cd... with no indices being summed over.
 * Indices may be transposed in any tensor.
 */
int tensor_souter_prod(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                          float  beta,          float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_douter_prod(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                         double  beta,         double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_couter_prod(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       complex float  beta,       complex float* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

int tensor_zouter_prod(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                                       const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,
                       complex double  beta,       complex double* C, gint_t ndim_C, const dim_t* len_C, const inc_t* stride_C, const char* idx_C);

/**
 * sum a tensor (presumably operated on in one or more ways) onto a second
 *
 * This form generalizes all of the unary operations trace, transpose, and replicate, which may be performed
 * in any combination.
 */
int tensor_ssum(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                   float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_dsum(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_csum(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_zsum(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

/**
 * Sum over (semi)diagonal elements of a tensor and sum onto a second
 *
 * The general form for a trace operation is ab...k*l*... -> ab... where k* denotes the index k appearing one or more times, etc. and where
 * the indices kl... will be summed (traced) over. Indices may be transposed, and multiple appearances
 * of the traced indices kl... need not appear together. Either set of indices may be empty, with the special case that when no indices
 * are traced over, the result is the same as transpose.
 */
int tensor_strace(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                     float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_dtrace(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                    double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_ctrace(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_ztrace(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                  complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

/**
 * Replicate a tensor and sum onto a second
 *
 * The general form for a replication operation is ab... -> ab...c*d*... where c* denotes the index c appearing one or more times.
 * Any indices may be transposed.
 */
int tensor_sreplicate(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                         float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_dreplicate(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                        double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_creplicate(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_zreplicate(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

/**
 * Transpose a tensor and sum onto a second
 *
 * The general form for a transposition operation is ab... -> P(ab...) where P is some permutation. Transposition may change
 * the order in which the elements of the tensor are physically stored.
 */
int tensor_stranspose(   float alpha, const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                         float  beta,          float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_dtranspose(  double alpha, const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                        double  beta,         double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_ctranspose(complex float alpha, const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex float  beta,       complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

int tensor_ztranspose(complex double alpha, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                      complex double  beta,       complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B);

/**
 * Return the dot product of two tensors
 */
int tensor_sdot(const    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const    float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,    float* val);

int tensor_ddot(const   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const   double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B,   double* val);

int tensor_cdot(const complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const complex float* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B, complex float* val);

int tensor_zdot(const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,
                const complex double* B, gint_t ndim_B, const dim_t* len_B, const inc_t* stride_B, const char* idx_B, complex double* val);

/**
 * Scale a tensor by a scalar
 */
int tensor_sscale(   float alpha,    float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A);

int tensor_dscale(  double alpha,   double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A);

int tensor_cscale(complex float alpha, complex float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A);

int tensor_zscale(complex double alpha, complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A);

/**
 * Return the reduction of a tensor, along with the corresponding index (as an offset from A) for MAX, MIN, MAX_ABS, and MIN_ABS reductions
 */
int tensor_sreduce(reduce_t op, const          float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,          float* val, inc_t* idx);

int tensor_dreduce(reduce_t op, const         double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A,         double* val, inc_t* idx);

int tensor_creduce(reduce_t op, const complex  float* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A, complex  float* val, inc_t* idx);

int tensor_zreduce(reduce_t op, const complex double* A, gint_t ndim_A, const dim_t* len_A, const inc_t* stride_A, const char* idx_A, complex double* val, inc_t* idx);

/**
 * Calculate the number of non-zero (and hence stored) elements in the given tensor.
 */
siz_t tensor_size(gint_t ndim, const dim_t* len, const inc_t* stride);

/**
 * Calculate the size of the buffer need for this tensor (in floating point words).
 */
siz_t tensor_storage_size(gint_t ndim, const dim_t* len, const inc_t* stride);

#endif
