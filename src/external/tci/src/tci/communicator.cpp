#include "tci/communicator.h"
#include "tci/parallel.h"

#if TCI_USE_TBB_THREADS

#include <tbb/tbb.h>

extern "C" void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                               tci_range range, tci_range_func func, void* payload)
{
    if (n == 1)
    {
        func(comm, 0, range.size, payload);
        return;
    }

    range.grain = TCI_MAX(range.grain, 1);
    uint64_t ngrain = (range.size+range.grain-1)/range.grain;

    tbb::task_group tg;

    for (unsigned idx = 0;idx < n;idx++)
    {
        tg.run(
        [&,idx]
        {
            uint64_t first = (idx*ngrain)/n;
            uint64_t last = ((idx+1)*ngrain)/n;

            func(comm, first*range.grain, TCI_MIN(last*range.grain, range.size), payload);
        });
    }

    tg.wait();
}

extern "C" void tci_distribute_2d(unsigned num, unsigned idx, tci_comm* comm,
                                  tci_range range_m, tci_range range_n,
                                  tci_range_2d_func func, void* payload)
{
    if (num == 1)
    {
        func(comm, 0, range_m.size, 0, range_n.size, payload);
        return;
    }

    unsigned m, n;
    tci_partition_2x2(num, range_m.size, num, range_n.size, num, &m, &n);

    range_m.grain = TCI_MAX(range_m.grain, 1);
    range_n.grain = TCI_MAX(range_n.grain, 1);
    uint64_t mgrain = (range_m.size+range_m.grain-1)/range_m.grain;
    uint64_t ngrain = (range_n.size+range_n.grain-1)/range_n.grain;

    tbb::task_group tg;

    for (unsigned idx_m = 0;idx_m < m;idx_m++)
    {
        for (unsigned idx_n = 0;idx_n < n;idx_n++)
        {
            tg.run(
            [&,idx_m,idx_n]
            {
                uint64_t mfirst = (idx_m*mgrain)/m;
                uint64_t nfirst = (idx_n*ngrain)/n;
                uint64_t mlast = ((idx_m+1)*mgrain)/m;
                uint64_t nlast = ((idx_n+1)*ngrain)/n;

                func(comm, mfirst*range_m.grain, TCI_MIN(mlast*range_m.grain, range_m.size),
                           nfirst*range_n.grain, TCI_MIN(nlast*range_n.grain, range_n.size), payload);
            });
        }
    }

    tg.wait();
}

#elif TCI_USE_PPL_THREADS

#include <ppl.h>

extern "C" void tci_distribute(unsigned n, unsigned idx, tci_comm* comm,
                               tci_range range, tci_range_func func, void* payload)
{
    (void)idx;

    if (n == 1)
    {
        func(comm, 0, range.size, payload);
        return;
    }

    range.grain = TCI_MAX(range.grain, 1);
    uint64_t ngrain = (range.size+range.grain-1)/range.grain;

    concurrency::parallel_for(uint64_t(), n,
    [&](uint64_t idx)
    {
        uint64_t first = (idx*ngrain)/n;
        uint64_t last = ((idx+1)*ngrain)/n;

        func(comm, first*range.grain,
             TCI_MIN(last*range.grain, range.size), payload);
    });
}

extern "C" void tci_distribute_2d(unsigned num, unsigned idx, tci_comm* comm,
                                  tci_range range_m, tci_range range_n,
                                  tci_range_2d_func func, void* payload)
{
    (void)idx;

    if (num == 1)
    {
        func(comm, 0, range_m.size, 0, range_n.size, payload);
        return;
    }

    unsigned m, n;
    tci_partition_2x2(num, range_m.size, num, range_n.size, num, &m, &n);

    range_m.grain = TCI_MAX(range_m.grain, 1);
    range_n.grain = TCI_MAX(range_n.grain, 1);
    uint64_t mgrain = (range_m.size+range_m.grain-1)/range_m.grain;
    uint64_t ngrain = (range_n.size+range_n.grain-1)/range_n.grain;

    concurrency::parallel_for(uint64_t(), m*n,
    [&](uint64_t idx)
    {
        unsigned idx_m = idx % m;
        unsigned idx_n = idx / m;

        uint64_t mfirst = (idx_m*mgrain)/m;
        uint64_t nfirst = (idx_n*ngrain)/n;
        uint64_t mlast = ((idx_m+1)*mgrain)/m;
        uint64_t nlast = ((idx_n+1)*ngrain)/n;

        func(comm, mfirst*range_m.grain,
                   TCI_MIN(mlast*range_m.grain, range_m.size),
                   nfirst*range_n.grain,
                   TCI_MIN(nlast*range_n.grain, range_n.size), payload);
    });
}

#endif
