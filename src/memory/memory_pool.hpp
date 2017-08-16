#ifndef _TBLIS_MEMORY_POOL_HPP_
#define _TBLIS_MEMORY_POOL_HPP_

#include <mutex>
#include <list>
#include <cstdlib>
#include <cstdio>

#if TBLIS_HAVE_HBWMALLOC_H
#include <hbwmalloc.h>
#endif

#include "util/thread.h"

namespace tblis
{

class MemoryPool
{
    public:
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

                template <typename T=void>
                T* get() { return static_cast<T*>(_ptr); }

                template <typename T=void>
                const T* get() const { return static_cast<const T*>(_ptr); }

                friend void swap(Block& a, Block& b)
                {
                    using std::swap;
                    swap(a._pool, b._pool);
                    swap(a._size, b._size);
                    swap(a._ptr, b._ptr);
                }

            protected:
                Block(MemoryPool* pool, size_t size, size_t alignment)
                : _pool(pool), _size(size),
                  _ptr(pool->acquire(size, alignment)) {}

                MemoryPool* _pool = nullptr;
                size_t _size = 0;
                void* _ptr = nullptr;
        };

        MemoryPool(size_t min_alignment=1) : _align(min_alignment) {}

        MemoryPool(const MemoryPool&) = delete;

        ~MemoryPool()
        {
            flush();
        }

        MemoryPool& operator=(const MemoryPool&) = delete;

        template <typename T>
        Block allocate(size_t num, size_t alignment=1)
        {
            return Block(this, num*sizeof(T),
                         std::max(alignment, std::alignment_of<T>::value));
        }

        void flush()
        {
            std::lock_guard<tci::mutex> guard(_lock);

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
            std::lock_guard<tci::mutex> guard(_lock);

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
                    reinterpret_cast<uintptr_t>(entry.first) % alignment == 0)
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
            std::lock_guard<tci::mutex> guard(_lock);

            TBLIS_ASSERT(ptr);
            _free_list.emplace_front(ptr, size);
        }

        std::list<std::pair<void*,size_t>> _free_list;
        tci::mutex _lock;
        size_t _align;
};

}

#endif
