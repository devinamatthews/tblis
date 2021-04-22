#ifndef _TCI_BARRIER_H_
#define _TCI_BARRIER_H_

#include "tci_global.h"

#include "yield.h"
#include "mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

#if TCI_USE_PTHREAD_BARRIER

typedef struct tci_barrier_node
{
    struct tci_barrier_node* parent;
    pthread_barrier_t barrier;
} tci_barrier_node;

#else

typedef struct tci_barrier_node
{
    struct tci_barrier_node* parent;
    unsigned nchildren;
    volatile unsigned step;
    volatile unsigned nwaiting;
} tci_barrier_node;

#endif

int tci_barrier_node_init(tci_barrier_node* barrier,
                          tci_barrier_node* parent,
                          unsigned nchildren);

int tci_barrier_node_destroy(tci_barrier_node* barrier);

int tci_barrier_node_wait(tci_barrier_node* barrier);

typedef struct tci_barrier
{
    union
    {
        tci_barrier_node* array;
        tci_barrier_node single;
    } barrier;
    unsigned nthread;
    unsigned group_size;
    int is_tree;
} tci_barrier;

int tci_barrier_is_tree(tci_barrier* barrier);

int tci_barrier_init(tci_barrier* barrier,
                     unsigned nthread, unsigned group_size);

int tci_barrier_destroy(tci_barrier* barrier);

int tci_barrier_wait(tci_barrier* barrier, unsigned tid);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TCI_DONT_USE_CXX11)

#include <system_error>

namespace tci
{

class barrier
{
    public:
        barrier(unsigned nthread, unsigned group_size=0)
        {
            int ret = tci_barrier_init(&_barrier, nthread, group_size);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        ~barrier() noexcept(false)
        {
            int ret = tci_barrier_destroy(&_barrier);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        barrier(const barrier&) = delete;

        barrier(barrier&) = default;

        barrier& operator=(const barrier&) = delete;

        barrier& operator=(barrier&) = default;

        void wait(unsigned tid)
        {
            int ret = tci_barrier_wait(&_barrier, tid);
            if (ret != 0) throw std::system_error(ret, std::system_category());
        }

        unsigned num_threads() const
        {
            return _barrier.nthread;
        }

        unsigned group_size() const
        {
            return _barrier.group_size;
        }

        operator tci_barrier*() { return &_barrier; }

        operator const tci_barrier*() const { return &_barrier; }

    protected:
        tci_barrier _barrier;
};

}

#endif

#endif
