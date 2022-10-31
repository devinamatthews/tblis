#ifndef _TCI_PIPELINE_H_
#define _TCI_PIPELINE_H_

#include "tci_global.h"

#include "yield.h"

#ifdef __cplusplus
extern "C" {
#endif

enum
{
    TCI_NOT_FILLED,
    TCI_DRAINING,
    TCI_FILLING,
    TCI_FILLED
};

typedef struct tci_pipeline
{
    void* buffer;
    size_t size;
    unsigned depth;
    unsigned last_drained;
    unsigned last_filled;
    volatile int state[1];
} tci_pipeline;

void tci_pipeline_init(tci_pipeline** p, unsigned depth, size_t size, void* buffer);

void tci_pipeline_destroy(tci_pipeline* p);

void* tci_pipeline_drain(tci_pipeline* p);

int tci_pipeline_trydrain(tci_pipeline* p, void** buffer);

void tci_pipeline_drained(tci_pipeline* p, void* buffer);

void* tci_pipeline_fill(tci_pipeline* p);

int tci_pipeline_tryfill(tci_pipeline* p, void** buffer);

void tci_pipeline_filled(tci_pipeline* p, void* buffer);

#ifdef __cplusplus
}

namespace tci
{

#include <utility>

namespace detail
{

template <typename T, typename Alloc>
class pipeline_base : private Alloc
{
    public:
        pipeline_base(unsigned depth, size_t size, Alloc alloc = Alloc())
        : Alloc(alloc)
        {
            tci_pipeline_init(&_p, depth, size*sizeof(T), allocate(depth*size));
        }

        ~pipeline_base()
        {
            deallocate(_p.buffer, _p.depth*_p.size/sizeof(T));
            tci_pipeline_destroy(_p);
        }

        pipeline_base(const pipeline_base&) = delete;

        pipeline_base& operator=(const pipeline_base&) = delete;

        operator tci_pipeline*() { return _p; }

        operator const tci_pipeline*() const { return _p; }

    protected:
        tci_pipeline* _p;
};

template <typename T>
class pipeline_base<T, void>
{
    public:
        pipeline_base(unsigned depth, size_t size, T* buffer)
        {
            tci_pipeline_init(&_p, depth, size*sizeof(T), buffer);
        }

        ~pipeline_base()
        {
            tci_pipeline_destroy(_p);
        }

        pipeline_base(const pipeline_base&) = delete;

        pipeline_base& operator=(const pipeline_base&) = delete;

        operator tci_pipeline*() { return _p; }

        operator const tci_pipeline*() const { return _p; }

    protected:
        tci_pipeline* _p;
};

}

template <typename T, typename Alloc=void>
class pipeline : public pipeline_base<T, Alloc>
{
    public:
        class drain_guard
        {
            public:
                drain_guard(pipeline& p)
                : _p(p), _buffer(p.drain()) {}

                drain_guard(const drain_guard&) = delete;

                drain_guard(drain_guard&& other)
                : _p(other._p), _buffer(other._buffer)
                {
                    other._p = nullptr;
                }

                drain_guard& operator=(const drain_guard&) = delete;

                drain_guard& operator=(drain_guard&& other)
                {
                    using std::swap;
                    swap(_p, other._p);
                    swap(_buffer, other._buffer);
                    return *this;
                }

                const T* buffer() const { return _buffer; }

                ~drain_guard()
                {
                    if (_p) _p->drained(_buffer);
                }

            protected:
                pipeline* _p;
                T* _buffer;
        };

        class fill_guard
        {
            public:
                fill_guard(pipeline& p)
                : _p(p), _buffer(p.fill()) {}

                fill_guard(const fill_guard&) = delete;

                fill_guard(fill_guard&& other)
                : _p(other._p), _buffer(other._buffer)
                {
                    other._p = nullptr;
                }

                fill_guard& operator=(const fill_guard&) = delete;

                fill_guard& operator=(fill_guard&& other)
                {
                    using std::swap;
                    swap(_p, other._p);
                    swap(_buffer, other._buffer);
                    return *this;
                }

                T* buffer() { return _buffer; }

                ~fill_guard()
                {
                    if (_p) _p->filled(_buffer);
                }

            protected:
                pipeline* _p;
                T* _buffer;
        };

        template <typename Dummy=Alloc, typename=typename std::enable_if<std::is_same<Alloc,void>::value>::type>
        pipeline(unsigned depth, size_t size, T* buffer)
        : pipeline_base(depth, size, buffer) {}

        template <typename Dummy=Alloc, typename=typename std::enable_if<!std::is_same<Alloc,void>::value>::type>
        pipeline(unsigned depth, size_t size, Alloc alloc = Alloc())
        : pipeline_base(depth, size, alloc) {}

        pipeline(const pipeline&) = delete;

        pipeline& operator=(const pipeline&) = delete;

        const T* drain()
        {
            return (T*)tci_pipeline_drain(*this);
        }

        std::pair<const T*,bool> try_drain()
        {
            const T* buffer;
            bool success = tci_pipeline_trydrain(*this, (void**)&buffer);
            return std::make_pair(buffer, success);
        }

        void drained(T* buffer)
        {
            tci_pipeline_drained(*this, buffer);
        }

        drain_guard guarded_drain()
        {
            return drain_guard(*this);
        }

        T* fill()
        {
            return (T*)tci_pipeline_fill(*this);
        }

        std::pair<T*,bool> try_fill()
        {
            T* buffer;
            bool success = tci_pipeline_tryfill(*this, &buffer);
            return std::make_pair(buffer, success);
        }

        void filled(T* buffer)
        {
            tci_pipeline_filled(*this, buffer);
        }

        fill_guard guarded_fill()
        {
            return fill_guard(*this);
        }
};

}

#endif

#endif
