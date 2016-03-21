#ifndef _TBLIS_TENSOR_CLASS_HPP_
#define _TBLIS_TENSOR_CLASS_HPP_

#include "tblis.hpp"

#include "util/util.hpp"
#include "util/tensor_check.hpp"

#include <algorithm>

namespace tblis
{

template <typename T> class Tensor;

namespace detail
{
    struct SortByIdx
    {
        const std::string& idx;

        SortByIdx(const std::string& idx) : idx(idx) {}

        bool operator()(gint_t i, gint_t j)
        {
            return idx[i] < idx[j];
        }
    };

    template <typename T>
    SortByIdx sortByIdx(const Tensor<T>& tensor, const std::string& idx)
    {
        return SortByIdx(idx);
    }

    template <typename T>
    struct SortByStride
    {
        const Tensor<T>& tensor;

        SortByStride(const Tensor<T>& tensor) : tensor(tensor) {}

        bool operator()(gint_t i, gint_t j)
        {
            return tensor.stride(i) < tensor.stride(j);
        }
    };

    template <typename T>
    SortByStride<T> sortByStride(const Tensor<T>& tensor, const std::string& idx)
    {
        return SortByStride<T>(tensor);
    }
}

template <typename T>
class Tensor
{
    public:
        typedef T type;
        typedef typename blis::real_type<T>::type real_type;

        Tensor(const Tensor& other)
        {
            create(other);
        }

        explicit Tensor(real_type r = real_type(), real_type i = real_type())
        {
            create(r, i);
        }

        explicit Tensor(type val)
        {
            create(real(val), imag(val));
        }

        template <typename len_type>
        Tensor(gint_t ndim, len_type len)
        {
            create(ndim, util::ptr(len));
        }

        template <typename len_type, typename stride_type>
        Tensor(gint_t ndim, len_type len, stride_type stride)
        {
            create(ndim, util::ptr(len), util::ptr(stride));
        }

        template <typename len_type>
        Tensor(gint_t ndim, len_type len, type* buf)
        {
            create(ndim, util::ptr(len), util::ptr(buf));
        }

        template <typename len_type, typename stride_type>
        Tensor(gint_t ndim, len_type len, type* buf, stride_type stride)
        {
            create(ndim, util::ptr(len), util::ptr(buf), util::ptr(stride));
        }

        template <typename len_type>
        Tensor(gint_t ndim, len_type len, const type* buf)
        {
            create(ndim, util::ptr(len), util::ptr(buf));
        }

        template <typename len_type, typename stride_type>
        Tensor(gint_t ndim_, len_type len, const type* buf, stride_type stride)
        {
            create(ndim_, util::ptr(len), util::ptr(buf), util::ptr(stride));
        }

        ~Tensor()
        {
            free();
        }

        Tensor& operator=(const Tensor& other)
        {
            reset(other);
            return *this;
        }

        Tensor& operator=(const T& x)
        {
            Iterator<> i(_len, _stride);
            for (T* ptr = _buf;i.next(ptr);) *ptr = x;
            return *this;
        }

        void reset(const Tensor& other)
        {
            if (&other == this) return;
            free();
            create(other);
        }

        void reset(real_type r = real_type(), real_type i = real_type())
        {
            free();
            create(r, i);
        }

        void reset(type val)
        {
            free();
            create(real(val), imag(val));
        }

        template <typename len_type>
        void reset(gint_t ndim, len_type len)
        {
            free();
            create(ndim, util::ptr(len));
        }

        template <typename len_type, typename stride_type>
        void reset(gint_t ndim, len_type len, stride_type stride)
        {
            free();
            create(ndim, util::ptr(len), util::ptr(stride));
        }

        template <typename len_type>
        void reset(gint_t ndim, len_type len, type* buf)
        {
            free();
            create(ndim, util::ptr(len), buf);
        }

        template <typename len_type, typename stride_type>
        void reset(gint_t ndim, len_type len, type* buf, stride_type stride)
        {
            free();
            create(ndim, util::ptr(len), util::ptr(buf), util::ptr(stride));
        }

        template <typename len_type>
        void reset(gint_t ndim, len_type len, const type* buf)
        {
            free();
            create(ndim, util::ptr(len), buf);
        }

        template <typename len_type, typename stride_type>
        void reset(gint_t ndim, len_type len, const type* buf, stride_type stride)
        {
            free();
            create(ndim, util::ptr(len), util::ptr(buf), util::ptr(stride));
        }

        gint_t dimension() const
        {
            return _ndim;
        }

        dim_t length(gint_t i) const
        {
            return _len[i];
        }

        const std::vector<dim_t>& lengths() const
        {
            return _len;
        }

        inc_t stride(gint_t i) const
        {
            return _stride[i];
        }

        const std::vector<inc_t>& strides() const
        {
            return _stride;
        }

        type* data()
        {
            #if TENSOR_ERROR_CHECKING
            if (_is_locked)
            {
                abort();
            }
            #endif
            return _buf;
        }

        const type* data() const
        {
            return _buf;
        }

        siz_t size() const
        {
            return _size;
        }

        bool isView() const
        {
            return _is_view;
        }

        bool isLocked() const
        {
            return _is_locked;
        }

        friend void Normalize(Tensor& A, std::string& idx_A)
        {
            std::vector<gint_t> inds_A = util::range<gint_t>(A._ndim);
            std::sort(inds_A.begin(), inds_A.end(), detail::sortByIdx(A, idx_A));

            std::string idx = idx_A;
            std::vector<dim_t> len = A._len;
            std::vector<inc_t> stride = A._stride;

            for (gint_t i = 0;i < A._ndim;i++)
            {
                    idx_A[i] = idx[inds_A[i]];
                    A._len[i] = len[inds_A[i]];
                    A._stride[i] = stride[inds_A[i]];
            }
        }

        friend void Normalize(Tensor& A , const std::string& idx_A,
                              Tensor& AN,       std::string& idx_AN)
        {
            LockedNormalize(A, idx_A, AN, idx_AN);
            AN._is_locked = A._is_locked;
        }

        friend void LockedNormalize(const Tensor& A , const std::string& idx_A,
                                          Tensor& AN,       std::string& idx_AN)
        {
            if (&A == &AN) abort();

            LockedView(A, AN);
            idx_AN = idx_A;
            Normalize(AN, idx_AN);
        }

        friend void Diagonal(Tensor& A , const std::string& idx_A,
                             Tensor& AD,       std::string& idx_AD)
        {
            LockedDiagonal(A, idx_A, AD, idx_AD);
            AD._is_locked = A._is_locked;
        }

        friend void LockedDiagonal(const Tensor& A , const std::string& idx_A,
                                         Tensor& AD,       std::string& idx_AD)
        {
            if (&A == &AD) abort();

            AD.free();
            AD._is_view = true;
            AD._is_locked = true;
            AD._buf = A._buf;
            AD._size = A._size;

            std::vector<gint_t> inds_A = util::range<gint_t>(A._ndim);
            std::sort(inds_A.begin(), inds_A.end(), detail::sortByIdx(A, idx_A));

            idx_AD.resize(A._ndim);
            AD._len.resize(A._ndim);
            AD._stride.resize(A._ndim);

            AD._ndim = 0;
            for (gint_t i = 0;i < A._ndim;i++)
            {
                if (i == 0 || idx_A[inds_A[i]] != idx_A[inds_A[i-1]])
                {
                    idx_AD[AD._ndim] = idx_A[inds_A[i]];
                    AD._len[AD._ndim] = A._len[inds_A[i]];
                    AD._stride[AD._ndim] = A._stride[inds_A[i]];
                    AD._ndim++;
                }
                else
                {
                    #if TENSOR_ERROR_CHECKING
                    if (AD._len[AD._ndim] != A._len[inds_A[i]])
                    {
                        abort();
                    }
                    #endif
                    AD._stride[AD._ndim] += A._stride[inds_A[i]];
                }
            }

            idx_AD.resize(AD._ndim);
            AD._len.resize(AD._ndim);
            AD._stride.resize(AD._ndim);
        }

        friend void View(Tensor& A, Tensor& B)
        {
            LockedView(A, B);
            B._is_locked = A._is_locked;
        }

        friend void LockedView(const Tensor& A, Tensor& B)
        {
            if (&A == &B)
            {
                if (!A._is_view) abort();
                return;
            }

            B.free();
            B._is_view   = true;
            B._is_locked = true;
            B._ndim      = A._ndim;
            B._len       = A._len;
            B._stride    = A._stride;
            B._size      = A._size;
            B._buf       = A._buf;
        }

        friend void View(Tensor& A, Tensor& B,
                         const std::vector<dim_t> off,
                         const std::vector<dim_t> len)
        {
            LockedView(A, B, off, len);
            B._is_locked = A._is_locked;
        }

        friend void LockedView(const Tensor& A, Tensor& B,
                               const std::vector<dim_t> off,
                               const std::vector<dim_t> len)
        {
            if (&A == &B) abort();

            if (off.size() != A._ndim ||
                len.size() != A._ndim)
            {
                abort();
            }

            LockedView(A, B);

            B._ndim = 0;
            for (gint_t i = 0;i < A._ndim;i++)
            {
                if (off[i] < 0 || off[i] >= A._len[i]) abort();
                if (len[i] < 0 || off[i]+len[i] > A._len[i]) abort();

                B._buf += off[i]*A.stride[i];

                if (len[i] > 0)
                {
                    B._len[B._ndim] = len[i];
                    B._stride[B._ndim] = A.stride[i];
                    B._ndim++;
                }
            }
        }

        friend void Partition(Tensor& A, Tensor& A0, Tensor& A1, gint_t dim, dim_t off)
        {
            LockedPartition(A, A0, A1, dim, off);
            A0._is_locked = A._is_locked;
            A1._is_locked = A._is_locked;
        }

        friend void LockedPartition(const Tensor& A, Tensor& A0, Tensor& A1, gint_t dim, dim_t off)
        {
            if (dim < 0 || dim >= A._ndim) abort();
            if (off < 0) abort();

            LockedView(A, A0);
            LockedView(A, A1);

            off = min(off, A._len[dim]);

            A0._len[dim]  = off;
            A1._len[dim] -= off;

            A1._buf += off*A._stride[dim];
        }

        friend void Unpartition(const Tensor& A0, const Tensor& A1, Tensor& A, gint_t dim)
        {
            if (A0._ndim != A1._ndim) abort();
            if (!A0._is_view || !A1._is_view) abort();
            if (A0._is_locked != A1._is_locked) abort();
            if (A0._stride != A1._stride) abort();
            if (dim < 0 || dim >= A0._ndim) abort();

            std::vector<dim_t> len_A1_ = A1._len;
            len_A1_[dim] = A0._len[dim];

            if (A0._len != len_A1_) abort();
            if (A1._buf != A0._buf+A0._len[dim]*A0._stride[dim]) abort();

            gint_t len = A0._len[dim]+A1._len[dim];

            LockedView(A0, A);

            A._len[dim] = len;
            A._is_locked = A0._is_locked;
        }

        friend void Slice(Tensor& A, Tensor& A0, Tensor& a1, Tensor& A2, gint_t dim, dim_t off)
        {
            LockedSlice(A, A0, a1, A2, dim, off);
            A0._is_locked = A._is_locked;
            a1._is_locked = A._is_locked;
            A2._is_locked = A._is_locked;
        }

        friend void LockedSlice(const Tensor& A, Tensor& A0, Tensor& a1, Tensor& A2, gint_t dim, dim_t off)
        {
            if (&A0 == &a1) abort();
            if (&A2 == &a1) abort();
            if (dim < 0 || dim >= A._ndim) abort();
            if (off < 0 || off >= A._len[dim]) abort();

            LockedView(A, A0);
            LockedView(A, a1);
            LockedView(A, A2);

            A0._len[dim] = off;
            a1._ndim--;
            a1._len.erase(a1._len.begin()+dim);
            a1._stride.erase(a1._stride.begin()+dim);
            A2._len[dim] -= off+1;

            a1._buf +=  off   *A0._stride[dim];
            A2._buf += (off+1)*A0._stride[dim];
        }

        friend void SliceFront(Tensor& A, Tensor& a0, Tensor& A1, gint_t dim)
        {
            LockedSliceFront(A, a0, A1, dim);
            a0._is_locked = A._is_locked;
            A1._is_locked = A._is_locked;
        }

        friend void LockedSliceFront(const Tensor& A, Tensor& a0, Tensor& A1, gint_t dim)
        {
            if (&A1 == &a0) abort();
            if (dim < 0 || dim >= A._ndim) abort();

            LockedView(A, a0);
            LockedView(A, A1);

            a0._ndim--;
            a0._len.erase(a0._len.begin()+dim);
            a0._stride.erase(a0._stride.begin()+dim);
            A1._len[dim]--;

            A1._buf += A1._stride[dim];
        }

        friend void SliceBack(Tensor& A, Tensor& A0, Tensor& a1, gint_t dim)
        {
            LockedSliceBack(A, A0, a1, dim);
            A0._is_locked = A._is_locked;
            a1._is_locked = A._is_locked;
        }

        friend void LockedSliceBack(const Tensor& A, Tensor& A0, Tensor& a1, gint_t dim)
        {
            if (&A0 == &a1) abort();
            if (dim < 0 || dim >= A._ndim) abort();

            LockedView(A, A0);
            LockedView(A, a1);

            A0._len[dim]--;
            a1._ndim--;
            a1._len.erase(a1._len.begin()+dim);
            a1._stride.erase(a1._stride.begin()+dim);

            a1._buf += (A0._len[dim]-1)*A0._stride[dim];
        }

        friend void Unslice(const Tensor& A0, const Tensor& a1, const Tensor& A2, Tensor& A, gint_t dim)
        {
            if (A0._ndim != a1._ndim+1) abort();
            if (A2._ndim != a1._ndim+1) abort();
            if (!A0._is_view || !a1._is_view || !A2._is_view) abort();
            if (A0._is_locked != a1._is_locked) abort();
            if (A2._is_locked != a1._is_locked) abort();
            if (A0._stride != A2._stride) abort();
            if (dim < 0 || dim >= A0._ndim) abort();

            std::vector<dim_t> len_A0_ = A0._len;
            len_A0_.erase(len_A0_.begin()+dim);

            std::vector<dim_t> len_A2_ = A2._len;
            len_A2_.erase(len_A2_.begin()+dim);

            std::vector<dim_t> stride_A0_ = A0._stride;
            stride_A0_.erase(stride_A0_.begin()+dim);

            if (len_A0_ != a1._len) abort();
            if (len_A2_ != a1._len) abort();
            if (stride_A0_ != a1._stride) abort();
            if (a1._buf != A0._buf+ A0._len[dim]   *A0._stride[dim]) abort();
            if (A2._buf != A0._buf+(A0._len[dim]+1)*A0._stride[dim]) abort();

            gint_t len = A0._len[dim]+A2._len[dim]+1;
            LockedView(A0, A);

            A._len[dim] = len;
            A._is_locked = A0._is_locked;
        }

        friend void UnsliceFront(const Tensor& a0, const Tensor& A1, Tensor& A, gint_t dim)
        {
            if (A1._ndim != a0._ndim+1) abort();
            if (!a0._is_view || !A1._is_view) abort();
            if (a0._is_locked != A1._is_locked) abort();
            if (dim < 0 || dim >= A1._ndim) abort();

            std::vector<dim_t> len_A1_ = A1._len;
            len_A1_.erase(len_A1_.begin()+dim);

            std::vector<dim_t> stride_A1_ = A1.stride;
            stride_A1_.erase(stride_A1_.begin()+dim);

            if (len_A1_ != a0._len) abort();
            if (stride_A1_ != a0._stride) abort();
            if (A1._buf != a0._buf+A1._stride[dim]) abort();

            LockedView(A1, A);

            A._len[dim]++;
            A._buf -= A1._stride[dim];
            A._is_locked = A1._is_locked;
        }

        friend void UnsliceBack(const Tensor& A0, const Tensor& a1, Tensor& A, gint_t dim)
        {
            if (A0._ndim != a1._ndim+1) abort();
            if (!A0._is_view || !a1._is_view) abort();
            if (A0._is_locked != a1._is_locked) abort();
            if (dim < 0 || dim >= A0._ndim) abort();

            std::vector<dim_t> len_A0_ = A0._len;
            len_A0_.erase(len_A0_.begin()+dim);

            std::vector<dim_t> stride_A0_ = A0._stride;
            stride_A0_.erase(stride_A0_.begin()+dim);

            if (len_A0_ != a1._len) abort();
            if (stride_A0_ != a1._stride) abort();
            if (a1._buf != A0._buf+A0._len[dim]*A0._stride[dim]) abort();

            LockedView(A0, A);

            A._len[dim]++;
            A._is_locked = A0._is_locked;
        }

        friend void Matricize(const Tensor& A, blis::Matrix<T>& AM, gint_t split)
        {
            if (split < 0 || split > A._ndim) abort();

            for (gint_t i = 1;i < A._ndim;i++)
            {
                if (i != split && A._stride[i] != A._stride[i-1]*A._len[i-1]) abort();
            }

            siz_t m = 1;
            for (gint_t i = 0;i < split;i++)
            {
                m *= A._len[i];
            }

            siz_t n = 1;
            for (gint_t i = split;i < A._ndim;i++)
            {
                n *= A._len[i];
            }

            inc_t rs, cs;

            if (A._ndim == 0)
            {
                rs = cs = 1;
            }
            else if (A._stride[0] == 1)
            {
                rs = (split ==       0 ? 1 : A._stride[    0]);
                cs = (split == A._ndim ? 1 : A._stride[split]);
            }
            else
            {
                rs = (split ==       0 ? 1 : A._stride[  split-1]);
                cs = (split == A._ndim ? 1 : A._stride[A._ndim-1]);
            }

            AM.reset(m, n, A._buf, rs, cs);
        }

        friend void Fold(Tensor& A , const std::string& idx_A,
                         Tensor& AF,       std::string& idx_AF)
        {
            LockedFold(A, idx_A, AF, idx_AF);
            AF._is_locked = A._is_locked;
        }

        friend void LockedFold(const Tensor& A , const std::string& idx_A,
                                     Tensor& AF,       std::string& idx_AF)
        {
            if (&A == &AF) abort();

            AF.free();
            AF._is_view = true;
            AF._is_locked = true;
            AF._buf = A._buf;
            AF._size = A._size;

            std::vector<gint_t> inds_A = util::range<gint_t>(A._ndim);
            std::sort(inds_A.begin(), inds_A.end(), detail::sortByStride(A, idx_A));

            idx_AF.resize(A._ndim);
            AF._len.resize(A._ndim);
            AF._stride.resize(A._ndim);

            AF._ndim = 0;
            for (gint_t i = 0;i < A._ndim;i++)
            {
                if (i == 0 || A._stride[inds_A[i]] != A._stride[inds_A[i-1]]*A._len[inds_A[i-1]])
                {
                    idx_AF[AF._ndim] = idx_A[inds_A[i]];
                    AF._len[AF._ndim] = A._len[inds_A[i]];
                    AF._stride[AF._ndim] = A._stride[inds_A[i]];
                    AF._ndim++;
                }
                else
                {
                    AF._len[AF._ndim] *= A._len[inds_A[i]];
                }
            }

            idx_AF.resize(AF._ndim);
            AF._len.resize(AF._ndim);
            AF._stride.resize(AF._ndim);
        }

        friend void Fold(Tensor& A, Tensor& B)
        {
            abort();
            //TODO
        }

        friend void Fold(Tensor& A, Tensor& B, Tensor& C)
        {
            abort();
            //TODO
        }

    protected:
        void create(const Tensor& other)
        {
            _is_view = other._is_view;
            _ndim = other._ndim;
            _len = other._len;
            _stride = other._stride;
            _size = other._size;

            if (!other._is_view)
            {
                _is_locked = false;
                _buf = (type*)bli_malloc(sizeof(type)*_size);
                std::copy_n(other._buf, _size, _buf);
            }
            else
            {
                _is_locked = other._is_locked;
                _buf = other._buf;
            }
        }

        void create(real_type r, real_type i)
        {
            create(0, (dim_t*)NULL, (inc_t*)NULL);
            std::copy(&r, &r+1, (real_type*)_buf);
            if (blis::is_complex<type>::value)
            {
                std::copy(&i, &i+1, (real_type*)_buf+1);
            }
        }

        void create(gint_t ndim, const dim_t* len)
        {
            _is_view = false;
            _is_locked = false;
            _ndim = ndim;
            _len.assign(len, len+ndim);
            _stride.resize(ndim, 1);

            _size = 1;
            for (gint_t i = 0;i < _ndim;i++)
            {
                _size *= _len[i];
                if (i > 0)
                {
                    _stride[i] = _stride[i-1]*_len[i-1];
                }
            }

            _buf = (type*)bli_malloc(sizeof(type)*_size);

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(_ndim, _len, _stride);
            #endif
        }

        void create(gint_t ndim, const dim_t* len, const inc_t* stride)
        {
            _is_view = false;
            _is_locked = false;
            _ndim = ndim;
            _len.assign(len, len+ndim);
            _stride.assign(stride, stride+ndim);

            _size = 1;
            for (gint_t i = 0;i < _ndim;i++)
            {
                _size += _stride[i]*(_len[i]-1);
            }

            _buf = (type*)bli_malloc(sizeof(type)*_size);

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(_ndim, _len, _stride);
            #endif
        }

        void create(gint_t ndim, const dim_t* len, type* buf)
        {
            create(ndim, len, static_cast<const type*>(buf));
            _is_locked = false;
        }

        void create(gint_t ndim, const dim_t* len, type* buf, const inc_t* stride)
        {
            create(ndim, len, static_cast<const type*>(buf), stride);
            _is_locked = false;
        }

        void create(gint_t ndim, const dim_t* len, const type* buf)
        {
            _is_view = true;
            _is_locked = true;
            _ndim = ndim;
            _len.assign(len, len+ndim);
            _stride.resize(ndim, 1);

            _size = 1;
            for (gint_t i = 0;i < _ndim;i++)
            {
                _size *= _len[i];
                if (i > 0)
                {
                    _stride[i] = _stride[i-1]*_len[i-1];
                }
            }

            _buf = const_cast<type*>(buf);

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(_ndim, _len, _buf, _stride);
            #endif
        }

        void create(gint_t ndim, const dim_t* len, const type* buf, const inc_t* stride)
        {
            _is_view = true;
            _is_locked = true;
            _ndim = ndim;
            _len.assign(len, len+ndim);
            _stride.assign(stride, stride+ndim);

            _size = 1;
            for (gint_t i = 0;i < _ndim;i++)
            {
                _size += _stride[i]*(_len[i]-1);
            }

            _buf = const_cast<type*>(buf);

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(_ndim, _len, _buf, _stride);
            #endif
        }

        void free()
        {
            if (!_is_view) bli_free(_buf);
        }

        bool _is_view;
        bool _is_locked;
        gint_t _ndim;
        std::vector<dim_t> _len;
        std::vector<inc_t> _stride;
        siz_t _size;
        type* _buf;
};

}

#endif
