#ifndef _TBLIS_MEMORY_POOL_HPP_
#define _TBLIS_MEMORY_POOL_HPP_

#include <memory>
#include <list>

#include "tblis_mutex.hpp"

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
                : _pool(other._pool), _num(other._num), _ptr(other._ptr)
                {
                    other._ptr = NULL;
                }

                ~Block()
                {
                    if (_ptr) _pool->release(_ptr, _num*sizeof(T));
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
                    swap(a._num, b._num);
                    swap(a._ptr, b._ptr);
                }

            protected:
                Block(MemoryPool* pool, size_t num, size_t alignment)
                : _pool(pool), _num(num),
                  _ptr((T*)pool->acquire(num*sizeof(T),
                                         std::max(alignment,
                                                  std::alignment_of<T>::value))) {}

                MemoryPool* _pool = NULL;
                size_t _num = 0;
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
            std::lock_guard<Mutex> guard(_lock);

            for (auto& entry : _free_list) free(entry.first);
            _free_list.clear();
        }

    protected:
        void* acquire(size_t size, size_t alignment)
        {
            std::lock_guard<Mutex> guard(_lock);

            alignment = std::max(alignment, _align);
            void* ptr = NULL;

            if (!_free_list.empty())
            {
                auto entry = _free_list.front();
                _free_list.pop_front();
                if (entry.second >= size) ptr = entry.first;
            }

            if (ptr == NULL)
            {
                if (posix_memalign(&ptr, alignment, size) != 0) abort();
            }

            return ptr;
        }

        void release(void* ptr, size_t size)
        {
            std::lock_guard<Mutex> guard(_lock);

            _free_list.emplace_front(ptr, size);
        }

        std::list<std::pair<void*,size_t>> _free_list;
        Mutex _lock;
        size_t _align;
};

}

#endif
