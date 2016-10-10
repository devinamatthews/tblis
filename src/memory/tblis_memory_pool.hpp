#ifndef _TBLIS_MEMORY_POOL_HPP_
#define _TBLIS_MEMORY_POOL_HPP_

#include "tblis.hpp"

#if TBLIS_HAVE_HBWMALLOC_H
#include <hbwmalloc.h>
#endif

namespace tblis
{

class MemoryPool
{
    public:
        template <typename T>
        class Block
        {
            friend class MemoryPool;

            public:
                Block() {}

                Block(const Block&) = delete;

                Block(Block&& other)
                : _pool(other._pool), _size(other._size), _ptr(other._ptr)
                {
                    other._ptr = NULL;
                }

                ~Block()
                {
                    if (_ptr) _pool->release(_ptr, _size);
                }

                Block& operator=(Block other)
                {
                    swap(*this, other);
                    return *this;
                }

                operator T*() { return _ptr; }

                operator const T*() const { return _ptr; }

                friend void swap(Block& a, Block& b)
                {
                    using std::swap;
                    swap(a._pool, b._pool);
                    swap(a._size, b._size);
                    swap(a._ptr, b._ptr);
                }

            protected:
                Block(MemoryPool* pool, size_t num, size_t alignment)
                : _pool(pool), _size(num*sizeof(T))
                {
                    alignment = std::max(alignment, std::alignment_of<T>::value);
                    _ptr = (T*)_pool->acquire(_size, alignment);
                }

                MemoryPool* _pool = NULL;
                size_t _size = 0;
                T* _ptr = NULL;
        };

        MemoryPool(size_t min_alignment=1) : _align(min_alignment) {};

        MemoryPool(const MemoryPool&) = delete;

        ~MemoryPool()
        {
            flush();
        }

        MemoryPool& operator=(const MemoryPool&) = delete;

        template <typename T>
        Block<T> allocate(size_t num, size_t alignment=1)
        {
            return Block<T>(this, num, alignment);
        }

        void flush()
        {
            std::lock_guard<mutex> guard(_lock);

            for (auto& entry : _free_list)
            {
                #if TBLIS_HAVE_HBWMALLOC_H
                hbw_free(entry.first);
                #else
                free(entry.first);
                #endif
            }
            _free_list.clear();
        }

    protected:
        void* acquire(size_t& size, size_t alignment)
        {
            std::lock_guard<mutex> guard(_lock);

            alignment = std::max(alignment, _align);
            void* ptr = NULL;

            if (!_free_list.empty())
            {
                auto entry = _free_list.front();
                _free_list.pop_front();

                /*
                 * If the region is big enough and properly aligned, use it.
                 * Otherwise, free it and allocate a new one.
                 */
                if (entry.second >= size &&
                    (uintptr_t)entry.first % alignment == 0)
                {
                    TBLIS_ASSERT(entry.first);
                    ptr = entry.first;
                    size = entry.second;
                }
                else
                {
                    #if TBLIS_HAVE_HBWMALLOC_H
                    hbw_free(entry.first);
                    #else
                    free(entry.first);
                    #endif
                }
            }

            if (ptr == NULL)
            {
                #if TBLIS_HAVE_HBWMALLOC_H
                int ret = hbw_posix_memalign(&ptr, alignment, size);
                #else
                int ret = posix_memalign(&ptr, alignment, size);
                #endif
                if (ret != 0)
                {
                    perror("posix_memalign");
                    abort();
                }
            }

            return ptr;
        }

        void release(void* ptr, size_t size)
        {
            std::lock_guard<mutex> guard(_lock);

            TBLIS_ASSERT(ptr);
            _free_list.emplace_front(ptr, size);
        }

        std::list<std::pair<void*,size_t>> _free_list;
        mutex _lock;
        size_t _align;
};

}

#endif
