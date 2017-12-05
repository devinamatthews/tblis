#include <stddef.h>
#include <tblis.h>
using namespace tblis;

static void _as_tensor(tblis_tensor *t, void *data, int dtype, int ndim,
                       ptrdiff_t *shape, ptrdiff_t *strides)
{
        switch (dtype) {
        case TYPE_SINGLE:
                tblis_init_tensor_s(t, ndim, shape, static_cast<float *>(data), strides);
                break;
        case TYPE_DOUBLE:
                tblis_init_tensor_d(t, ndim, shape, static_cast<double *>(data), strides);
                break;
        case TYPE_SCOMPLEX:
                tblis_init_tensor_c(t, ndim, shape, static_cast<scomplex *>(data), strides);
                break;
        case TYPE_DCOMPLEX:
                tblis_init_tensor_z(t, ndim, shape, static_cast<dcomplex *>(data), strides);
                break;
        }
}

extern "C" {
void as_einsum(void *data_A, int ndim_A, ptrdiff_t *shape_A, ptrdiff_t *strides_A, char *descr_A,
               void *data_B, int ndim_B, ptrdiff_t *shape_B, ptrdiff_t *strides_B, char *descr_B,
               void *data_C, int ndim_C, ptrdiff_t *shape_C, ptrdiff_t *strides_C, char *descr_C,
               int dtype)
{
        tblis_tensor A, B, C;
        _as_tensor(&A, data_A, dtype, ndim_A, shape_A, strides_A);
        _as_tensor(&B, data_B, dtype, ndim_B, shape_B, strides_B);
        _as_tensor(&C, data_C, dtype, ndim_C, shape_C, strides_C);

        tblis_tensor_mult(NULL, NULL, &A, descr_A, &B, descr_B, &C, descr_C);
}
}
