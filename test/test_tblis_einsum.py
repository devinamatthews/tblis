import unittest
import numpy
import tblis_einsum
from tblis_einsum import einsum

class KnownValues(unittest.TestCase):
    def test_d_d(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_c_c(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32)
        b = numpy.random.random((2,4,5,7)).astype(numpy.float32)
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-5)

    def test_c_d(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32) + 0j
        b = numpy.random.random((2,4,5,7)).astype(numpy.float32)
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-5)

    def test_d_z(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7)) + 0j
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_z_z(self):
        a = numpy.random.random((7,1,3,4)) + 0j
        b = numpy.random.random((2,4,5,7)) + 0j
        c0 = numpy.einsum('abcd,fdea->cebf', a, b)
        c1 = einsum('abcd,fdea->cebf', a, b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_dslice(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_dslice1(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[:4].copy(), b[:,:,:,2:6])
        c1 = einsum('abcd,fdea->cebf', a[:4].copy(), b[:,:,:,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_d(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[:,:,1:3,:], b)
        c1 = einsum('abcd,fdea->cebf', a[:,:,1:3,:], b)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_dslice(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[:,:,1:3], b[:,:,:2,:])
        c1 = einsum('abcd,fdea->cebf', a[:,:,1:3], b[:,:,:2,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_dslice_dslice1(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[2:6,:,:1], b[:,:,1:3,2:6])
        c1 = einsum('abcd,fdea->cebf', a[2:6,:,:1], b[:,:,1:3,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_d_cslice(self):
        a = numpy.random.random((7,1,3,4))
        b = numpy.random.random((2,4,5,7)).astype(numpy.float32)
        c0 = numpy.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_z_cslice(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32) + 0j
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        c1 = einsum('abcd,fdea->cebf', a, b[:,:,1:3,:])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_cslice_dslice(self):
        a = numpy.random.random((7,1,3,4)).astype(numpy.float32) + 0j
        b = numpy.random.random((2,4,5,7))
        c0 = numpy.einsum('abcd,fdea->cebf', a[2:6], b[:,:,1:3,2:6])
        c1 = einsum('abcd,fdea->cebf', a[2:6], b[:,:,1:3,2:6])
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-14)

    def test_3operands(self):
        a = numpy.random.random((7,1,3,4)) + 1j
        b = numpy.random.random((2,4,5,7))
        c = numpy.random.random((2,8,3,6))
        c0 = numpy.einsum('abcd,fdea,ficj->iebj', a, b, c)
        c1 = einsum('abcd,fdea,ficj->iebj', a, b, c)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_1operand(self):
        a = numpy.random.random((4,1,3,4)) + 1j
        c0 = numpy.einsum('abca->bc', a)
        c1 = einsum('abca->bc', a)
        self.assertTrue(c0.dtype == c1.dtype)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_wrong_dimension(self):
        a = numpy.random.random((5,1,3,4))
        b = numpy.random.random((2,4,5,7))
        self.assertRaises(ValueError, einsum, 'abcd,fdea->cebf', a, b)

    def test_contraction1(self):
        a = numpy.random.random((5,2,3,2))
        c0 = numpy.einsum('ijkj->k', a)
        c1 = einsum('ijkj->k', a)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction2(self):
        a = numpy.random.random((5,2,3,2))
        b = numpy.random.random((2,4,7))+1j
        c0 = numpy.einsum('ijkj,jlp->pk', a, b)
        c1 = einsum('ijkj,jlp->pk', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction3(self):
        a = numpy.random.random((5,2,3,2))
        b = numpy.random.random((2,4,7))+1j
        c0 = numpy.einsum('ijkj,jlp->jpk', a, b)
        c1 = einsum('ijkj,jlp->jpk', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contraction4(self):
        a = numpy.random.random((5,2,3,2))
        b = numpy.random.random((2,4,7))
        c0 = numpy.einsum('...jkj,jlp->...jp', a, b)
        c1 = einsum('...jkj,jlp->...jp', a, b)
        self.assertTrue(abs(c0-c1).max() < 1e-13)

    def test_contract(self):
        a = numpy.random.random((5,4,6))
        b = numpy.random.random((4,9,6))

        c1 = numpy.ones((9,5), dtype=numpy.complex128)
        c0 = tblis_einsum._contract('ijk,jlk->li', a, b, out=c1, alpha=.5j, beta=.2)
        c1 = numpy.ones((9,5), dtype=numpy.complex128)
        c1 = c1*.2 + numpy.einsum('ijk,jlk->li', a, b)*.5j
        self.assertTrue(abs(c0-c1).max() < 1e-13)


if __name__ == '__main__':
    unittest.main()
