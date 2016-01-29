#ifndef _TENSOR_BLIS___MEMORY_HPP_
#define _TENSOR_BLIS___MEMORY_HPP_

#include "blis++.hpp"

namespace blis
{

template <typename T, typename Allocator=std::allocator<T> >
class Memory
{
    public:
        typedef T type;

    private:
        struct _mem_s : Allocator
        {
            type* _base;
            _mem_s(Allocator alloc) : Allocator(alloc), _base(NULL) {}
        } _mem;
        type* _ptr;
        siz_t _size;

    public:
        Memory(const Memory&) = delete;

        Memory(Memory&&) = default;

        Memory& operator=(const Memory&) = delete;

        Memory& operator=(Memory&&) = default;

        explicit Memory(siz_t size, Allocator alloc = Allocator())
        : _mem(alloc)
        {
            reset(size);
        }

        explicit Memory(Allocator alloc = Allocator())
        : _mem(alloc)
        {
            free();
        }

        explicit Memory(type* ptr)
        : _mem(Allocator())
        {
            reset(ptr);
        }

        ~Memory()
        {
            free();
        }

        void reset(siz_t size = 0)
        {
            free();
            if (size > 0)
            {
                _mem._base = _mem.allocate(size);
                _size = size;
                _ptr = _mem._base;
            }
        }

        void reset(type* ptr)
        {
            free();
            _ptr = ptr;
        }

        void free()
        {
            if (_mem._base) _mem.deallocate(_mem._base, _size);
            _mem._base = NULL;
            _size = 0;
            _ptr = NULL;
        }

        Memory& operator+=(inc_t x)
        {
            _ptr += x;
            return *this;
        }

        Memory& operator-=(inc_t x)
        {
            _ptr -= x;
            return *this;
        }

        operator type*() { return _ptr; }

        operator const type*() const { return _ptr; }
};

template <typename T>
class PooledMemory : private mem_t
{
    public:
        typedef T type;

    private:
        packbuf_t _packbuf;

        void init()
        {
            memset(static_cast<mem_t*>(this), 0, sizeof(mem_t));
        }

    public:
        PooledMemory(const PooledMemory&) = delete;

        PooledMemory& operator=(const PooledMemory&) = delete;

        explicit PooledMemory(packbuf_t packbuf) : _packbuf(packbuf)
        {
            init();
        }

        explicit PooledMemory(siz_t size, packbuf_t packbuf) : _packbuf(packbuf)
        {
            init();
            reset(size);
        }

        ~PooledMemory()
        {
            free();
        }

        void reset() {}

        void reset(siz_t size)
        {
            if (bli_mem_is_unalloc(this))
            {
                bli_mem_acquire_m(size, _packbuf, this);
            }
            else if (size > bli_mem_size(this))
            {
                bli_mem_release(this);
                bli_mem_acquire_m(size, _packbuf, this);
            }
        }

        void free()
        {
            if (bli_mem_is_alloc(this)) bli_mem_release(this);
        }

        operator type*() { return (type*)bli_mem_buffer(this); }

        operator const type*() const { return (type*)bli_mem_buffer(this); }
};

}

#endif
