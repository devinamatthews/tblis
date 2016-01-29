#ifndef _TENSOR_CORE_TENSOR_CLASS_HPP_
#define _TENSOR_CORE_TENSOR_CLASS_HPP_

#include "config.h"

#include "blis++/matrix.hpp"

#include "util/util.hpp"
#include "util/tensor_check.hpp"
#include "util/iterator.hpp"

#include <algorithm>

namespace tensor
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
            return tensor.stride[i] < tensor.stride[j];
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
        Tensor(gint_t ndim_, len_type len_)
        {
            create(ndim_, util::ptr(len_));
        }

        template <typename len_type, typename stride_type>
        Tensor(gint_t ndim_, len_type len_, stride_type stride_)
        {
            create(ndim_, util::ptr(len_), util::ptr(stride_));
        }

        template <typename len_type>
        Tensor(gint_t ndim_, len_type len_, type* buf_)
        {
            create(ndim_, util::ptr(len_), util::ptr(buf_));
        }

        template <typename len_type, typename stride_type>
        Tensor(gint_t ndim_, len_type len_, type* buf_, stride_type stride_)
        {
            create(ndim_, util::ptr(len_), util::ptr(buf_), util::ptr(stride_));
        }

        template <typename len_type>
        Tensor(gint_t ndim_, len_type len_, const type* buf_)
        {
            create(ndim_, util::ptr(len_), util::ptr(buf_));
        }

        template <typename len_type, typename stride_type>
        Tensor(gint_t ndim_, len_type len_, const type* buf_, stride_type stride_)
        {
            create(ndim_, util::ptr(len_), util::ptr(buf_), util::ptr(stride_));
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
            util::Iterator i(len, stride);
            for (T* ptr = buf;i.nextIteration(ptr);) *ptr = x;
            return *this;
        }

        void conjugate()
        {
            if (!blis::is_complex<T>::value) return;
            util::Iterator i(len, stride);
            for (T* ptr = buf;i.nextIteration(ptr);) *ptr = blis::conj(*ptr);
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
        void reset(gint_t ndim_, len_type len_)
        {
            free();
            create(ndim_, util::ptr(len_));
        }

        template <typename len_type, typename stride_type>
        void reset(gint_t ndim_, len_type len_, stride_type stride_)
        {
            free();
            create(ndim_, util::ptr(len_), util::ptr(stride_));
        }

        template <typename len_type>
        void reset(gint_t ndim_, len_type len_, type* buf_)
        {
            free();
            create(ndim_, util::ptr(len_), buf_);
        }

        template <typename len_type, typename stride_type>
        void reset(gint_t ndim_, len_type len_, type* buf_, stride_type stride_)
        {
            free();
            create(ndim_, util::ptr(len_), util::ptr(buf_), util::ptr(stride_));
        }

        template <typename len_type>
        void reset(gint_t ndim_, len_type len_, const type* buf_)
        {
            free();
            create(ndim_, util::ptr(len_), buf_);
        }

        template <typename len_type, typename stride_type>
        void reset(gint_t ndim_, len_type len_, const type* buf_, stride_type stride_)
        {
            free();
            create(ndim_, util::ptr(len_), util::ptr(buf_), util::ptr(stride_));
        }

        gint_t getDimension() const
        {
            return ndim;
        }

        dim_t getLength(gint_t i) const
        {
            return len[i];
        }

        const std::vector<dim_t>& getLengths() const
        {
            return len;
        }

        inc_t getStride(gint_t i) const
        {
            return stride[i];
        }

        const std::vector<inc_t>& getStrides() const
        {
            return stride;
        }

        type* getData()
        {
            #if TENSOR_ERROR_CHECKING
            if (is_locked)
            {
                abort();
            }
            #endif
            return buf;
        }

        const type* getData() const
        {
            return buf;
        }

        siz_t getDataSize() const
        {
            return size;
        }

        operator type*()
        {
            return buf;
        }

        operator const type*() const
        {
            return buf;
        }

        bool isView() const
        {
            return is_view;
        }

        bool isLocked() const
        {
            return is_locked;
        }

        friend void Normalize(Tensor& A, std::string& idx_A)
        {
            std::vector<gint_t> inds_A = util::range<gint_t>(A.getDimension());
            std::sort(inds_A.begin(), inds_A.end(), detail::sortByIdx(A, idx_A));

            std::string idx = idx_A;
            std::vector<dim_t> len = A.len;
            std::vector<inc_t> stride = A.stride;

            for (gint_t i = 0;i < A.ndim;i++)
            {
                    idx_A[i] = idx[inds_A[i]];
                    A.len[i] = len[inds_A[i]];
                    A.stride[i] = stride[inds_A[i]];
            }
        }

        friend void Normalize(Tensor& A , const std::string& idx_A,
                              Tensor& AN,       std::string& idx_AN)
        {
            LockedNormalize(A, idx_A, AN, idx_AN);
            AN.is_locked = A.is_locked;
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
            AD.is_locked = A.is_locked;
        }

        friend void LockedDiagonal(const Tensor& A , const std::string& idx_A,
                                         Tensor& AD,       std::string& idx_AD)
        {
            if (&A == &AD) abort();

            AD.free();
            AD.is_view = true;
            AD.is_locked = true;
            AD.buf = A.buf;
            AD.size = A.size;

            std::vector<gint_t> inds_A = util::range<gint_t>(A.getDimension());
            std::sort(inds_A.begin(), inds_A.end(), detail::sortByIdx(A, idx_A));

            idx_AD.resize(A.ndim);
            AD.len.resize(A.ndim);
            AD.stride.resize(A.ndim);

            AD.ndim = 0;
            for (gint_t i = 0;i < A.ndim;i++)
            {
                if (i == 0 || idx_A[inds_A[i]] != idx_A[inds_A[i-1]])
                {
                    idx_AD[AD.ndim] = idx_A[inds_A[i]];
                    AD.len[AD.ndim] = A.len[inds_A[i]];
                    AD.stride[AD.ndim] = A.stride[inds_A[i]];
                    AD.ndim++;
                }
                else
                {
                    #if TENSOR_ERROR_CHECKING
                    if (AD.len[AD.ndim] != A.len[inds_A[i]])
                    {
                        abort();
                    }
                    #endif
                    AD.stride[AD.ndim] += A.stride[inds_A[i]];
                }
            }

            idx_AD.resize(AD.ndim);
            AD.len.resize(AD.ndim);
            AD.stride.resize(AD.ndim);
        }

        friend void View(Tensor& A, Tensor& B)
        {
            LockedView(A, B);
            B.is_locked = A.is_locked;
        }

        friend void LockedView(const Tensor& A, Tensor& B)
        {
            if (&A == &B)
            {
                if (!A.is_view) abort();
                return;
            }

            B.free();
            B.is_view   = true;
            B.is_locked = true;
            B.ndim      = A.ndim;
            B.len       = A.len;
            B.stride    = A.stride;
            B.size      = A.size;
            B.buf       = A.buf;
        }

        friend void View(Tensor& A, Tensor& B,
                         const std::vector<dim_t> off,
                         const std::vector<dim_t> len)
        {
            LockedView(A, B, off, len);
            B.is_locked = A.is_locked;
        }

        friend void LockedView(const Tensor& A, Tensor& B,
                               const std::vector<dim_t> off,
                               const std::vector<dim_t> len)
        {
            if (&A == &B) abort();

            if (off.size() != A.ndim ||
                len.size() != A.ndim)
            {
                abort();
            }

            LockedView(A, B);

            B.ndim = 0;
            for (gint_t i = 0;i < A.ndim;i++)
            {
                if (off[i] < 0 || off[i] >= A.len[i]) abort();
                if (len[i] < 0 || off[i]+len[i] > A.len[i]) abort();

                B.buf += off[i]*A.stride[i];

                if (len[i] > 0)
                {
                    B.len[B.ndim] = len[i];
                    B.stride[B.ndim] = A.stride[i];
                    B.ndim++;
                }
            }
        }

        friend void Partition(Tensor& A, Tensor& A0, Tensor& A1, gint_t dim, dim_t off)
        {
            LockedPartition(A, A0, A1, dim, off);
            A0.is_locked = A.is_locked;
            A1.is_locked = A.is_locked;
        }

        friend void LockedPartition(const Tensor& A, Tensor& A0, Tensor& A1, gint_t dim, dim_t off)
        {
            if (dim < 0 || dim >= A.ndim) abort();
            if (off < 0) abort();

            LockedView(A, A0);
            LockedView(A, A1);

            off = min(off, A.len[dim]);

            A0.len[dim]  = off;
            A1.len[dim] -= off;

            A1.buf += off*A.stride[dim];
        }

        friend void Unpartition(const Tensor& A0, const Tensor& A1, Tensor& A, gint_t dim)
        {
            if (A0.ndim != A1.ndim) abort();
            if (!A0.is_view || !A1.is_view) abort();
            if (A0.is_locked != A1.is_locked) abort();
            if (A0.stride != A1.stride) abort();
            if (dim < 0 || dim >= A0.ndim) abort();

            std::vector<dim_t> len_A1_ = A1.len;
            len_A1_[dim] = A0.len[dim];

            if (A0.len != len_A1_) abort();
            if (A1.buf != A0.buf+A0.len[dim]*A0.stride[dim]) abort();

            gint_t len = A0.len[dim]+A1.len[dim];

            LockedView(A0, A);

            A.len[dim] = len;
            A.is_locked = A0.is_locked;
        }

        friend void Slice(Tensor& A, Tensor& A0, Tensor& a1, Tensor& A2, gint_t dim, dim_t off)
        {
            LockedSlice(A, A0, a1, A2, dim, off);
            A0.is_locked = A.is_locked;
            a1.is_locked = A.is_locked;
            A2.is_locked = A.is_locked;
        }

        friend void LockedSlice(const Tensor& A, Tensor& A0, Tensor& a1, Tensor& A2, gint_t dim, dim_t off)
        {
            if (&A0 == &a1) abort();
            if (&A2 == &a1) abort();
            if (dim < 0 || dim >= A.ndim) abort();
            if (off < 0 || off >= A.len[dim]) abort();

            LockedView(A, A0);
            LockedView(A, a1);
            LockedView(A, A2);

            A0.len[dim] = off;
            a1.ndim--;
            a1.len.erase(a1.len.begin()+dim);
            a1.stride.erase(a1.stride.begin()+dim);
            A2.len[dim] -= off+1;

            a1.buf +=  off   *A0.stride[dim];
            A2.buf += (off+1)*A0.stride[dim];
        }

        friend void SliceFront(Tensor& A, Tensor& a0, Tensor& A1, gint_t dim)
        {
            LockedSliceFront(A, a0, A1, dim);
            a0.is_locked = A.is_locked;
            A1.is_locked = A.is_locked;
        }

        friend void LockedSliceFront(const Tensor& A, Tensor& a0, Tensor& A1, gint_t dim)
        {
            if (&A1 == &a0) abort();
            if (dim < 0 || dim >= A.ndim) abort();

            LockedView(A, a0);
            LockedView(A, A1);

            a0.ndim--;
            a0.len.erase(a0.len.begin()+dim);
            a0.stride.erase(a0.stride.begin()+dim);
            A1.len[dim]--;

            A1.buf += A1.stride[dim];
        }

        friend void SliceBack(Tensor& A, Tensor& A0, Tensor& a1, gint_t dim)
        {
            LockedSliceBack(A, A0, a1, dim);
            A0.is_locked = A.is_locked;
            a1.is_locked = A.is_locked;
        }

        friend void LockedSliceBack(const Tensor& A, Tensor& A0, Tensor& a1, gint_t dim)
        {
            if (&A0 == &a1) abort();
            if (dim < 0 || dim >= A.ndim) abort();

            LockedView(A, A0);
            LockedView(A, a1);

            A0.len[dim]--;
            a1.ndim--;
            a1.len.erase(a1.len.begin()+dim);
            a1.stride.erase(a1.stride.begin()+dim);

            a1.buf += (A0.len[dim]-1)*A0.stride[dim];
        }

        friend void Unslice(const Tensor& A0, const Tensor& a1, const Tensor& A2, Tensor& A, gint_t dim)
        {
            if (A0.ndim != a1.ndim+1) abort();
            if (A2.ndim != a1.ndim+1) abort();
            if (!A0.is_view || !a1.is_view || !A2.is_view) abort();
            if (A0.is_locked != a1.is_locked) abort();
            if (A2.is_locked != a1.is_locked) abort();
            if (A0.stride != A2.stride) abort();
            if (dim < 0 || dim >= A0.ndim) abort();

            std::vector<dim_t> len_A0_ = A0.len;
            len_A0_.erase(len_A0_.begin()+dim);

            std::vector<dim_t> len_A2_ = A2.len;
            len_A2_.erase(len_A2_.begin()+dim);

            std::vector<dim_t> stride_A0_ = A0.stride;
            stride_A0_.erase(stride_A0_.begin()+dim);

            if (len_A0_ != a1.len) abort();
            if (len_A2_ != a1.len) abort();
            if (stride_A0_ != a1.stride) abort();
            if (a1.buf != A0.buf+ A0.len[dim]   *A0.stride[dim]) abort();
            if (A2.buf != A0.buf+(A0.len[dim]+1)*A0.stride[dim]) abort();

            gint_t len = A0.len[dim]+A2.len[dim]+1;
            LockedView(A0, A);

            A.len[dim] = len;
            A.is_locked = A0.is_locked;
        }

        friend void UnsliceFront(const Tensor& a0, const Tensor& A1, Tensor& A, gint_t dim)
        {
            if (A1.ndim != a0.ndim+1) abort();
            if (!a0.is_view || !A1.is_view) abort();
            if (a0.is_locked != A1.is_locked) abort();
            if (dim < 0 || dim >= A1.ndim) abort();

            std::vector<dim_t> len_A1_ = A1.len;
            len_A1_.erase(len_A1_.begin()+dim);

            std::vector<dim_t> stride_A1_ = A1.stride;
            stride_A1_.erase(stride_A1_.begin()+dim);

            if (len_A1_ != a0.len) abort();
            if (stride_A1_ != a0.stride) abort();
            if (A1.buf != a0.buf+A1.stride[dim]) abort();

            LockedView(A1, A);

            A.len[dim]++;
            A.buf -= A1.stride[dim];
            A.is_locked = A1.is_locked;
        }

        friend void UnsliceBack(const Tensor& A0, const Tensor& a1, Tensor& A, gint_t dim)
        {
            if (A0.ndim != a1.ndim+1) abort();
            if (!A0.is_view || !a1.is_view) abort();
            if (A0.is_locked != a1.is_locked) abort();
            if (dim < 0 || dim >= A0.ndim) abort();

            std::vector<dim_t> len_A0_ = A0.len;
            len_A0_.erase(len_A0_.begin()+dim);

            std::vector<dim_t> stride_A0_ = A0.stride;
            stride_A0_.erase(stride_A0_.begin()+dim);

            if (len_A0_ != a1.len) abort();
            if (stride_A0_ != a1.stride) abort();
            if (a1.buf != A0.buf+A0.len[dim]*A0.stride[dim]) abort();

            LockedView(A0, A);

            A.len[dim]++;
            A.is_locked = A0.is_locked;
        }

        friend void Matricize(const Tensor& A, blis::Matrix<T>& AM, gint_t split)
        {
            if (split < 0 || split > A.ndim) abort();

            for (gint_t i = 1;i < A.ndim;i++)
            {
                if (i != split && A.stride[i] != A.stride[i-1]*A.len[i-1]) abort();
            }

            siz_t m = 1;
            for (gint_t i = 0;i < split;i++)
            {
                m *= A.len[i];
            }

            siz_t n = 1;
            for (gint_t i = split;i < A.ndim;i++)
            {
                n *= A.len[i];
            }

            inc_t rs, cs;

            if (A.ndim == 0)
            {
                rs = cs = 1;
            }
            else if (A.stride[0] == 1)
            {
                rs = (split ==      0 ? 1 : A.stride[0]);
                cs = (split == A.ndim ? 1 : A.stride[split]);
            }
            else
            {
                rs = (split ==      0 ? 1 : A.stride[ split-1]);
                cs = (split == A.ndim ? 1 : A.stride[A.ndim-1]);
            }

            AM.reset(m, n, A.buf, rs, cs);
        }

        friend void Fold(Tensor& A , const std::string& idx_A,
                         Tensor& AF,       std::string& idx_AF)
        {
            LockedFold(A, idx_A, AF, idx_AF);
            AF.is_locked = A.is_locked;
        }

        friend void LockedFold(const Tensor& A , const std::string& idx_A,
                                     Tensor& AF,       std::string& idx_AF)
        {
            if (&A == &AF) abort();

            AF.free();
            AF.is_view = true;
            AF.is_locked = true;
            AF.buf = A.buf;
            AF.buf = A.buf;
            AF.size = A.size;

            std::vector<gint_t> inds_A = util::range<gint_t>(A.getDimension());
            std::sort(inds_A.begin(), inds_A.end(), detail::sortByStride(A, idx_A));

            idx_AF.resize(A.ndim);
            AF.len.resize(A.ndim);
            AF.stride.resize(A.ndim);

            AF.ndim = 0;
            for (gint_t i = 0;i < A.ndim;i++)
            {
                if (i == 0 || A.stride[inds_A[i]] != A.stride[inds_A[i-1]]*A.len[inds_A[i-1]])
                {
                    idx_AF[AF.ndim] = idx_A[inds_A[i]];
                    AF.len[AF.ndim] = A.len[inds_A[i]];
                    AF.stride[AF.ndim] = A.stride[inds_A[i]];
                    AF.ndim++;
                }
                else
                {
                    AF.len[AF.ndim] *= A.len[inds_A[i]];
                }
            }

            idx_AF.resize(AF.ndim);
            AF.len.resize(AF.ndim);
            AF.stride.resize(AF.ndim);
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
            is_view = other.is_view;
            ndim = other.ndim;
            len = other.len;
            stride = other.stride;
            size = other.size;

            if (!other.is_view)
            {
                is_locked = false;
                buf = (type*)bli_malloc(sizeof(type)*size);
                std::copy(other.buf, other.buf+size, buf);
            }
            else
            {
                is_locked = other.is_locked;
                buf = other.buf;
            }
        }

        void create(real_type r, real_type i)
        {
            create(0, (dim_t*)NULL, (inc_t*)NULL);
            std::copy(&r, &r+1, (real_type*)buf);
            if (blis::is_complex<type>::value)
            {
                std::copy(&i, &i+1, (real_type*)buf+1);
            }
        }

        void create(gint_t ndim_, const dim_t* len_)
        {
            is_view = false;
            is_locked = false;
            ndim = ndim_;
            len.assign(len_, len_+ndim);
            stride.resize(ndim, 1);

            size = 1;
            for (gint_t i = 0;i < ndim;i++)
            {
                size *= len[i];
                if (i > 0)
                {
                    stride[i] = stride[i-1]*len[i-1];
                }
            }

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(ndim, len, stride);
            #endif

            buf = (type*)bli_malloc(sizeof(type)*size);
        }

        void create(gint_t ndim_, const dim_t* len_, const inc_t* stride_)
        {
            is_view = false;
            is_locked = false;
            ndim = ndim_;
            len.assign(len_, len_+ndim);
            stride.assign(stride_, stride_+ndim);

            size = 1;
            for (gint_t i = 0;i < ndim;i++)
            {
                size += stride[i]*(len[i]-1);
            }

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(ndim, len, stride);
            #endif

            buf = (type*)bli_malloc(sizeof(type)*size);
        }

        void create(gint_t ndim_, const dim_t* len_, type* buf_)
        {
            create(ndim_, len_, static_cast<const type*>(buf_));
            is_locked = false;
        }

        void create(gint_t ndim_, const dim_t* len_, type* buf_, const inc_t* stride_)
        {
            create(ndim_, len_, static_cast<const type*>(buf_), stride_);
            is_locked = false;
        }

        void create(gint_t ndim_, const dim_t* len_, const type* buf_)
        {
            is_view = true;
            is_locked = true;
            ndim = ndim_;
            len.assign(len_, len_+ndim);
            stride.resize(ndim, 1);

            size = 1;
            for (gint_t i = 0;i < ndim;i++)
            {
                size *= len[i];
                if (i > 0)
                {
                    stride[i] = stride[i-1]*len[i-1];
                }
            }

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(ndim, len, buf_, stride);
            #endif

            buf = const_cast<type*>(buf_);
        }

        void create(gint_t ndim_, const dim_t* len_, const type* buf_, const inc_t* stride_)
        {
            is_view = true;
            is_locked = true;
            ndim = ndim_;
            len.assign(len_, len_+ndim);
            stride.assign(stride_, stride_+ndim);

            size = 1;
            for (gint_t i = 0;i < ndim;i++)
            {
                size += stride[i]*(len[i]-1);
            }

            #if TENSOR_ERROR_CHECKING
            util::check_tensor(ndim, len, buf_, stride);
            #endif

            buf = const_cast<type*>(buf_);
        }

        void free()
        {
            if (!is_view) bli_free(buf);
        }

        bool is_view;
        bool is_locked;
        gint_t ndim;
        std::vector<dim_t> len;
        std::vector<inc_t> stride;
        siz_t size;
        type* buf;
};

}

#endif
