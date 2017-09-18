import re
import ctypes
import numpy

libtblis = ctypes.CDLL('libtblis_itrf.so')

libtblis.as_einsum.restype = None
libtblis.as_einsum.argtypes = (
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    numpy.ctypeslib.ndpointer(), ctypes.c_int,
    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int
)

tblis_dtype = {
    numpy.dtype(numpy.float32)    : 0,
    numpy.dtype(numpy.double)     : 1,
    numpy.dtype(numpy.complex64)  : 2,
    numpy.dtype(numpy.complex128) : 3,
}

def _contract(subscripts, *tensors, **kwargs):
    subscripts = subscripts.replace(' ','')
    sub_idx = re.split(',|->', subscripts)
    indices  = ''.join(sub_idx)
    c_dtype = numpy.result_type(*tensors)
    if (not (',' in subscripts and '->' in subscripts) or
        any(indices.count(x)>2 for x in set(indices)) or
        numpy.issubdtype(c_dtype, numpy.integer)):
        return numpy.einsum(subscripts, *tensors)

    a = numpy.asarray(tensors[0], dtype=c_dtype)
    b = numpy.asarray(tensors[1], dtype=c_dtype)

    a_shape = a.shape
    b_shape = b.shape
    a_descr, b_descr, c_descr = sub_idx
    a_shape_dic = dict(zip(a_descr, a_shape))
    b_shape_dic = dict(zip(b_descr, b_shape))
    if any(a_shape_dic[x] != b_shape_dic[x]
           for x in set(a_descr).intersection(b_descr)):
        raise ValueError('operands dimension error for "%s" : %s %s'
                         % (subscripts, a_shape, b_shape))

    ab_shape_dic = a_shape_dic
    ab_shape_dic.update(b_shape_dic)
    c_shape = tuple([ab_shape_dic[x] for x in c_descr])

    out = getattr(kwargs, 'out', None)
    if out is None:
        c = numpy.zeros(c_shape, dtype=c_dtype)
    else:
        assert(out.dtype == c_dtype)
        assert(out.shape == c_shape)
        c = out
        c[:] = 0

    a_shape = (ctypes.c_size_t*a.ndim)(*a_shape)
    b_shape = (ctypes.c_size_t*a.ndim)(*b_shape)
    c_shape = (ctypes.c_size_t*a.ndim)(*c_shape)

    nbytes = c_dtype.itemsize
    a_strides = (ctypes.c_size_t*a.ndim)(*[x//nbytes for x in a.strides])
    b_strides = (ctypes.c_size_t*a.ndim)(*[x//nbytes for x in b.strides])
    c_strides = (ctypes.c_size_t*a.ndim)(*[x//nbytes for x in c.strides])

    libtblis.as_einsum(a, a.ndim, a_shape, a_strides, a_descr,
                       b, b.ndim, b_shape, b_strides, b_descr,
                       c, c.ndim, c_shape, c_strides, c_descr,
                       tblis_dtype[c_dtype])
    return c

def einsum(subscripts, *tensors, **kwargs):
    if len(tensors) <= 2:
        return _contract(subscripts, *tensors, **kwargs)
    else:
        sub_idx = subscripts.split(',')
        res_idx = ''.join(set(sub_idx[0]).symmetric_difference(sub_idx[1]))
        script1 = '->'.join((','.join(sub_idx[:2]), res_idx))
        t0 = _contract(script1, *tensors[:2])
        subscripts = ','.join([res_idx] + sub_idx[2:])
        tensors = [t0] + list(tensors[2:])
        return einsum(subscripts, *tensors, **kwargs)

