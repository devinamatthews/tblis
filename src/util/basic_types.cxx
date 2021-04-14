#include "basic_types.h"

namespace tblis
{

extern "C"
{

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

void tblis_init_tensor_scaled_s(tblis_tensor* t, float scalar,
                                int ndim, len_type* len, float* data,
                                stride_type* stride)
{
    t->type = TYPE_SINGLE;
    t->scalar.type = TYPE_SINGLE;
    t->conj = 0;
    t->scalar.data.s = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_d(tblis_tensor* t, double scalar,
                                int ndim, len_type* len, double* data,
                                stride_type* stride)
{
    t->type = TYPE_DOUBLE;
    t->scalar.type = TYPE_DOUBLE;
    t->conj = 0;
    t->scalar.data.d = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_c(tblis_tensor* t, scomplex scalar,
                                int ndim, len_type* len, scomplex* data,
                                stride_type* stride)
{
    t->type = TYPE_SCOMPLEX;
    t->scalar.type = TYPE_SCOMPLEX;
    t->conj = 0;
    t->scalar.data.c = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_scaled_z(tblis_tensor* t, dcomplex scalar,
                                int ndim, len_type* len, dcomplex* data,
                                stride_type* stride)
{
    t->type = TYPE_DCOMPLEX;
    t->scalar.type = TYPE_DCOMPLEX;
    t->conj = 0;
    t->scalar.data.z = scalar;
    t->data = data;
    t->ndim = ndim;
    t->len = len;
    t->stride = stride;
}

void tblis_init_tensor_s(tblis_tensor* t,
                         int ndim, len_type* len, float* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_s(t, 1.0f, ndim, len, data, stride);
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
    tblis_init_tensor_scaled_c(t, {1.0f, 0.0f}, ndim, len, data, stride);
}

void tblis_init_tensor_z(tblis_tensor* t,
                         int ndim, len_type* len, dcomplex* data,
                         stride_type* stride)
{
    tblis_init_tensor_scaled_z(t, {1.0, 0.0}, ndim, len, data, stride);
}

}

label_vector idx(const std::string& from, label_vector&& to)
{
    constexpr auto N = sizeof(label_type);

    union
    {
        label_type label;
        char chars[N];
    };

    label = 0;
    size_t i = 0;
    for (char c : from)
    {
        if (c == ',')
        {
            TBLIS_ASSERT(i == 0, "Malformed index string: %s", from.c_str());
            to.push_back(label);
            label = 0;
            i = 0;
            continue;
        }

        TBLIS_ASSERT(i >= N, "Label name too long: %s", from.c_str());
        chars[i++] = c;
    }

    TBLIS_ASSERT(i == 0, "Malformed index string: %s", from.c_str());
    to.push_back(label);

    return std::move(to);
}

}
