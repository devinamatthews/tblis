'''
A Python interface to mimic numpy.einsum
'''

import sys
import re
import ctypes
import numpy

if (sys.platform.startswith('linux') or
    sys.platform.startswith('gnukfreebsd')):
    so_ext = '.so'
elif sys.platform.startswith('darwin'):
    so_ext = '.dylib'
elif sys.platform.startswith('win'):
    so_ext = '.dll'
else:
    raise ImportError('Unsupported platform')

libtblis = ctypes.CDLL('libtblis'+so_ext)

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

numpy_einsum = numpy.einsum

def _contract(subscripts, *tensors, **kwargs):
    sub_idx = re.split(',|->', subscripts)
    indices  = ''.join(sub_idx)
    c_dtype = getattr(kwargs, 'dtype', numpy.result_type(*tensors))
    if ('...' in subscripts or
        not (numpy.issubdtype(c_dtype, numpy.float) or
             numpy.issubdtype(c_dtype, numpy.complex))):
        return numpy_einsum(subscripts, *tensors)

    if '->' not in subscripts:
        # Find chararacters which appear only once in the subscripts for c_descr
        for x in set(indices):
            if indices.count(x) > 1:
                indices = indices.replace(x, '')
        sub_idx += [indices]

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
    b_shape = (ctypes.c_size_t*b.ndim)(*b_shape)
    c_shape = (ctypes.c_size_t*c.ndim)(*c_shape)

    nbytes = c_dtype.itemsize
    a_strides = (ctypes.c_size_t*a.ndim)(*[x//nbytes for x in a.strides])
    b_strides = (ctypes.c_size_t*b.ndim)(*[x//nbytes for x in b.strides])
    c_strides = (ctypes.c_size_t*c.ndim)(*[x//nbytes for x in c.strides])

    libtblis.as_einsum(a, a.ndim, a_shape, a_strides, a_descr,
                       b, b.ndim, b_shape, b_strides, b_descr,
                       c, c.ndim, c_shape, c_strides, c_descr,
                       tblis_dtype[c_dtype])
    return c

def einsum(subscripts, *tensors, **kwargs):
    subscripts = subscripts.replace(' ','')
    order = getattr(kwargs, 'order', None)
    if len(tensors) <= 1:
        out = numpy_einsum(subscripts, *tensors, **kwargs)
    elif len(tensors) <= 2:
        out = _contract(subscripts, *tensors, **kwargs)
        out = numpy.asarray(out, order=order)
    else:
        sub_idx = subscripts.split(',', 2)
        res_idx = ''.join(set(sub_idx[0]).symmetric_difference(sub_idx[1]))
        script0 = sub_idx[0] + ',' + sub_idx[1] + '->' + res_idx
        subscripts = res_idx + ',' + sub_idx[2]
        tensors = [_contract(script0, *tensors[:2])] + list(tensors[2:])
        out = einsum(subscripts, *tensors, **kwargs)
    return out

