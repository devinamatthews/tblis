#include <tblis/base/types.h>

using namespace tblis;

extern "C"
{

void tblis_init_scalar_s(tblis_scalar* s, float value)
{
    s->reset(value);
}

void tblis_init_scalar_d(tblis_scalar* s, double value)
{
    s->reset(value);
}

void tblis_init_scalar_c(tblis_scalar* s, scomplex value)
{
    s->reset(value);
}

void tblis_init_scalar_z(tblis_scalar* s, dcomplex value)
{
    s->reset(value);
}

void tblis_init_tensor_scaled_s(tblis_tensor* t, float scalar,
                                int ndim, len_type* len, float* data,
                                stride_type* stride)
{
    t->scalar.reset(scalar);
    t->conj = 0;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_d(tblis_tensor* t, double scalar,
                                int ndim, len_type* len, double* data,
                                stride_type* stride)
{
    t->scalar.reset(scalar);
    t->conj = 0;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_c(tblis_tensor* t, scomplex scalar,
                                int ndim, len_type* len, scomplex* data,
                                stride_type* stride)
{
    t->scalar.reset(scalar);
    t->conj = 0;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_z(tblis_tensor* t, dcomplex scalar,
                                int ndim, len_type* len, dcomplex* data,
                                stride_type* stride)
{
    t->scalar.reset(scalar);
    t->conj = 0;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_s(tblis_tensor* t,
                         int ndim, len_type* len, float* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_s(t, 1.0, ndim, len, data, stride);
}

void tblis_init_tensor_d(tblis_tensor* t,
                         int ndim, len_type* len, double* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_d(t, 1.0, ndim, len, data, stride);
}

void tblis_init_tensor_c(tblis_tensor* t,
                         int ndim, len_type* len, scomplex* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_c(t, 10, ndim, len, data, stride);
}

void tblis_init_tensor_z(tblis_tensor* t,
                         int ndim, len_type* len, dcomplex* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_z(t, 1.0, ndim, len, data, stride);
}

}
