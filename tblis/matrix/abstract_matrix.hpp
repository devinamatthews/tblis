#ifndef _TBLIS_ABSTRACT_MATRIX_HPP_
#define _TBLIS_ABSTRACT_MATRIX_HPP_

#include <memory>

#include <tblis/base/types.h>
#include <tblis/base/thread.h>

#include <tblis/internal/configs.hpp>
#include <tblis/internal/alignment.hpp>
#include <tblis/internal/memory_pool.hpp>

namespace tblis
{

struct matrix_implementation
{
    virtual ~matrix_implementation() {}
};

class abstract_matrix
{
    private:
        scalar scale_;
        bool conj_ = false;
        bool row_major_ = false;
        bool transposed_ = false;
        std::array<len_type,2> tot_len_ = {};
        std::array<len_type,2> cur_len_ = {};
        std::array<len_type,2> off_ = {};
        struct pack_buffer : MemoryPool::Block
        {
            pack_buffer() {}

            pack_buffer(const pack_buffer&) : MemoryPool::Block() {}

            pack_buffer(pack_buffer&&) = default;

            pack_buffer& operator=(const pack_buffer&) = delete;

            pack_buffer& operator=(pack_buffer&&) = default;

            pack_buffer& operator=(MemoryPool::Block&& other)
            {
                MemoryPool::Block::operator=(std::move(other));
                return *this;
            }
        } pack_buffer_;
        char* pack_ptr_ = nullptr;
        stride_type pack_size_ = 0;

    protected:
        std::shared_ptr<matrix_implementation> impl_;
        abstract_matrix* (*clone_)(const abstract_matrix&) = nullptr;
        abstract_matrix (*pack_)(abstract_matrix&,
                                 const communicator& comm, const config& cfg,
                                 int mat, MemoryPool& pool) = nullptr;
        void (*gemm_)(abstract_matrix&, const communicator& comm, const config& cfg,
                      MemoryPool& pool, const abstract_matrix& A,
                      const abstract_matrix& B) = nullptr;

        abstract_matrix(const scalar& scalar, bool conj,
                        bool row_major, len_type m, len_type n,
                        std::shared_ptr<matrix_implementation>&& impl)
        : scale_(scalar), conj_(conj), row_major_(row_major),
          tot_len_{m, n}, cur_len_{m, n}, impl_(std::move(impl)) {}

        char* get_buffer(const communicator& comm, stride_type nelem, MemoryPool& pool)
        {
            if (pack_size_ < nelem)
            {
                if (comm.master())
                {
                    pack_buffer_ = pool.allocate(nelem*type_size[type()], 4096);
                    pack_ptr_ = pack_buffer_.get<char>();
                }

                pack_size_ = nelem;
                comm.broadcast_value(pack_ptr_);
            }

            return pack_ptr_;
        }

        char* get_buffer() const
        {
            return pack_ptr_;
        }

        stride_type get_buffer_size() const
        {
            return pack_size_;
        }

        void set_buffer(char* buf, stride_type size)
        {
            pack_ptr_ = buf;
            pack_size_ = size;
        }

        void reset(bool row_major, len_type m, len_type n)
        {
            row_major_ = row_major;
            transposed_ = false;
            tot_len_[0] = cur_len_[0] = m;
            tot_len_[1] = cur_len_[1] = n;
            off_ = {};
        }

    public:
        abstract_matrix(type_t type)
        : scale_(1, type) {}

        type_t type() const
        {
            return scale_.type;
        }

        len_type length(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return cur_len_[dim^transposed_];
        }

        len_type length(int dim, len_type len)
        {
            len_type old_len = length(dim);
            shift_and_resize(dim, 0, len);
            return old_len;
        }

        len_type offset(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return off_[dim^transposed_];
        }

        len_type offset(int dim, len_type off)
        {
            len_type old_off = offset(dim);
            shift_and_resize(dim, off-old_off, length(dim));
            return old_off;
        }

        void shift(int dim, len_type n)
        {
            shift_and_resize(dim, n, length(dim));
        }

        void shift_and_resize(int dim, len_type n, len_type len)
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            off_[dim^transposed_] += n;
            cur_len_[dim^transposed_] = len;
            TBLIS_ASSERT(offset(dim) >= 0);
            TBLIS_ASSERT(offset(dim)+length(dim) <= tot_len_[dim^transposed_]);
        }

        void transpose()
        {
            transposed_ = !transposed_;
        }

        abstract_matrix* clone() const
        {
            TBLIS_ASSERT(clone_, "This matrix cannot be cloned.");
            return clone_(*this);
        }

        abstract_matrix pack(const communicator& comm, const config& cfg,
                             int mat, MemoryPool& pool)
        {
            TBLIS_ASSERT(pack_, "This matrix cannot be packed.");
            return pack_(*this, comm, cfg, mat, pool);
        }

        void gemm(const communicator& comm, const config& cfg,
                  MemoryPool& pool, const abstract_matrix& A,
                  const abstract_matrix& B)
        {
            TBLIS_ASSERT(gemm_, "This matrix cannot be multiplied into.");
            gemm_(*this, comm, cfg, pool, A, B);
        }

        void set_scaled()
        {
            scale_.set(1.0);
            conj_ = false;
        }

        bool row_major() const
        {
            return row_major_;
        }

        const scalar& scale() const
        {
            return scale_;
        }

        bool conj() const
        {
            return conj_;
        }

        bool transposed() const
        {
            return transposed_;
        }
};

template <typename Derived>
static abstract_matrix* do_clone(const abstract_matrix& self)
{
    return new Derived(static_cast<const Derived&>(self));
}

template <typename Derived, typename Impl=void>
class abstract_matrix_adapter : public abstract_matrix
{
    protected:
        using abstract_matrix::impl_;

        struct matrix_implementation_wrapper : matrix_implementation
        {
            Impl impl;

            template <typename... Args>
            matrix_implementation_wrapper(Args&&... args)
            : impl(std::forward<Args>(args)...) {}
        };

        const Impl& impl() const
        {
            return static_cast<matrix_implementation_wrapper*>(impl_.get())->impl;
        }

        template <typename... Args>
        abstract_matrix_adapter(const tblis_scalar& alpha, bool conj,
                                bool row_major, len_type m, len_type n,
                                Args&&... args)
        : abstract_matrix(alpha, conj, row_major, m, n,
                          std::make_shared<matrix_implementation_wrapper>(std::forward<Args>(args)...))
        {
            clone_ = do_clone<Derived>;
        }
};

template <typename Derived>
class abstract_matrix_adapter<Derived, void>
: public abstract_matrix
{
    protected:
        abstract_matrix_adapter(const tblis_scalar& alpha, bool conj,
                                bool row_major, len_type m, len_type n)
        : abstract_matrix(alpha, conj, row_major, m, n, {})
        {
            clone_ = do_clone<Derived>;
        }
};

}

#endif
