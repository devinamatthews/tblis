#include "basic_types.h"

#ifdef __cplusplus
namespace tblis {
extern "C" {
#endif

void tblis_init_scalar_s(tblis_scalar* s, float value)
{
    s->type = TYPE_SINGLE;
    s->data.s = value;
}

void tblis_init_scalar_d(tblis_scalar* s, double value)
{
    s->type = TYPE_DOUBLE;
    s->data.d = value;
}

void tblis_init_scalar_c(tblis_scalar* s, scomplex value)
{
    s->type = TYPE_SCOMPLEX;
    s->data.c = value;
}

void tblis_init_scalar_z(tblis_scalar* s, dcomplex value)
{
    s->type = TYPE_DCOMPLEX;
    s->data.z = value;
}

void tblis_init_vector_scaled_s(tblis_vector* v, float scalar,
                                len_type n, float* data,stride_type inc)
{
    v->type = TYPE_SINGLE;
    v->conj = 0;
    v->scalar.s = scalar;
    v->data = data;
    v->n = n;
    v->inc = inc;
}

void tblis_init_vector_scaled_d(tblis_vector* v, double scalar,
                                len_type n, double* data,stride_type inc)
{
    v->type = TYPE_DOUBLE;
    v->conj = 0;
    v->scalar.d = scalar;
    v->data = data;
    v->n = n;
    v->inc = inc;
}

void tblis_init_vector_scaled_c(tblis_vector* v, scomplex scalar,
                                len_type n, scomplex* data,stride_type inc)
{
    v->type = TYPE_SCOMPLEX;
    v->conj = 0;
    v->scalar.c = scalar;
    v->data = data;
    v->n = n;
    v->inc = inc;
}

void tblis_init_vector_scaled_z(tblis_vector* v, dcomplex scalar,
                                len_type n, dcomplex* data,stride_type inc)
{
    v->type = TYPE_DCOMPLEX;
    v->conj = 0;
    v->scalar.z = scalar;
    v->data = data;
    v->n = n;
    v->inc = inc;
}

void tblis_init_vector_s(tblis_vector* v,
                         len_type n, float* data,stride_type inc)
{
    tblis_init_vector_scaled_s(v, 1.0f, n, data, inc);
}

void tblis_init_vector_d(tblis_vector* v,
                         len_type n, double* data,stride_type inc)
{
    tblis_init_vector_scaled_d(v, 1.0, n, data, inc);
}

void tblis_init_vector_c(tblis_vector* v,
                         len_type n, scomplex* data,stride_type inc)
{
    tblis_init_vector_scaled_c(v, {1.0f, 0.0f}, n, data, inc);
}

void tblis_init_vector_z(tblis_vector* v,
                         len_type n, dcomplex* data,stride_type inc)
{
    tblis_init_vector_scaled_z(v, {1.0, 0.0}, n, data, inc);
}

void tblis_init_matrix_scaled_s(tblis_matrix* mat, float scalar,
                                len_type m, len_type n, float* data,
                                stride_type rs, stride_type cs)
{
    mat->type = TYPE_SINGLE;
    mat->conj = 0;
    mat->scalar.s = scalar;
    mat->data = data;
    mat->m = m;
    mat->n = n;
    mat->rs = rs;
    mat->cs = cs;
}

void tblis_init_matrix_scaled_d(tblis_matrix* mat, double scalar,
                                len_type m, len_type n, double* data,
                                stride_type rs, stride_type cs)
{
    mat->type = TYPE_DOUBLE;
    mat->conj = 0;
    mat->scalar.d = scalar;
    mat->data = data;
    mat->m = m;
    mat->n = n;
    mat->rs = rs;
    mat->cs = cs;
}

void tblis_init_matrix_scaled_c(tblis_matrix* mat, scomplex scalar,
                                len_type m, len_type n, scomplex* data,
                                stride_type rs, stride_type cs)
{
    mat->type = TYPE_SCOMPLEX;
    mat->conj = 0;
    mat->scalar.c = scalar;
    mat->data = data;
    mat->m = m;
    mat->n = n;
    mat->rs = rs;
    mat->cs = cs;
}

void tblis_init_matrix_scaled_z(tblis_matrix* mat, dcomplex scalar,
                                len_type m, len_type n, dcomplex* data,
                                stride_type rs, stride_type cs)
{
    mat->type = TYPE_DCOMPLEX;
    mat->conj = 0;
    mat->scalar.z = scalar;
    mat->data = data;
    mat->m = m;
    mat->n = n;
    mat->rs = rs;
    mat->cs = cs;
}

void tblis_init_matrix_s(tblis_matrix* mat,
                         len_type m, len_type n, float* data,
                         stride_type rs, stride_type cs)
{
    tblis_init_matrix_scaled_s(mat, 1.0f, m, n, data, rs, cs);
}

void tblis_init_matrix_d(tblis_matrix* mat,
                         len_type m, len_type n, double* data,
                         stride_type rs, stride_type cs)
{
    tblis_init_matrix_scaled_d(mat, 1.0, m, n, data, rs, cs);
}

void tblis_init_matrix_c(tblis_matrix* mat,
                         len_type m, len_type n, scomplex* data,
                         stride_type rs, stride_type cs)
{
    tblis_init_matrix_scaled_c(mat, {1.0f, 0.0f}, m, n, data, rs, cs);
}

void tblis_init_matrix_z(tblis_matrix* mat,
                         len_type m, len_type n, dcomplex* data,
                         stride_type rs, stride_type cs)
{
    tblis_init_matrix_scaled_z(mat, {1.0, 0.0}, m, n, data, rs, cs);
}

void tblis_init_tensor_scaled_s(tblis_tensor* t, float scalar,
                                unsigned ndim, len_type* len, float* data,
                                stride_type* stride)
{
    t->type = TYPE_SINGLE;
    t->conj = 0;
    t->scalar.s = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_d(tblis_tensor* t, double scalar,
                                unsigned ndim, len_type* len, double* data,
                                stride_type* stride)
{
    t->type = TYPE_DOUBLE;
    t->conj = 0;
    t->scalar.d = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_c(tblis_tensor* t, scomplex scalar,
                                unsigned ndim, len_type* len, scomplex* data,
                                stride_type* stride)
{
    t->type = TYPE_SCOMPLEX;
    t->conj = 0;
    t->scalar.c = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_z(tblis_tensor* t, dcomplex scalar,
                                unsigned ndim, len_type* len, dcomplex* data,
                                stride_type* stride)
{
    t->type = TYPE_DCOMPLEX;
    t->conj = 0;
    t->scalar.z = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_s(tblis_tensor* t,
                         unsigned ndim, len_type* len, float* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_s(t, 1.0f, ndim, len, data, stride);
}

void tblis_init_tensor_d(tblis_tensor* t,
                         unsigned ndim, len_type* len, double* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_d(t, 1.0, ndim, len, data, stride);
}

void tblis_init_tensor_c(tblis_tensor* t,
                         unsigned ndim, len_type* len, scomplex* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_c(t, {1.0f, 0.0f}, ndim, len, data, stride);
}

void tblis_init_tensor_z(tblis_tensor* t,
                         unsigned ndim, len_type* len, dcomplex* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_z(t, {1.0, 0.0}, ndim, len, data, stride);
}

#ifdef __cplusplus
}
}
#endif
