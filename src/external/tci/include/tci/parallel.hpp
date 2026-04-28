#ifndef _TCI_PARALLEL_HPP_
#define _TCI_PARALLEL_HPP_

#include "tci/parallel.h"
#include "tci/communicator.hpp"

#include <system_error>
#include <tuple>
#include <utility>

namespace tci
{

template <typename Body>
void parallelize(Body&& body, unsigned nthread, unsigned arity=0)
{
    tci_parallelize(
        [](tci_comm* comm, void* data)
        {
            Body body = *static_cast<Body*>(data);
            body(*reinterpret_cast<const communicator*>(comm));
        },
        static_cast<void*>(&body), nthread, arity);
}

class prime_factorization
{
    public:
        prime_factorization(unsigned n)
        {
            tci_prime_factorization(n, &_factors);
        }

        unsigned next()
        {
            return tci_next_prime_factor(&_factors);
        }

    protected:
        tci_prime_factors _factors;
};

inline std::pair<unsigned,unsigned>
partition_2x2(unsigned nthreads, uint64_t work1, uint64_t work2)
{
    unsigned nt1, nt2;
    tci_partition_2x2(nthreads, work1, nthreads, work2, nthreads, &nt1, &nt2);
    return std::make_pair(nt1, nt2);
}

inline std::pair<unsigned,unsigned>
partition_2x2(unsigned nthreads,
              uint64_t work1, unsigned max1,
              uint64_t work2, unsigned max2)
{
    unsigned nt1, nt2;
    tci_partition_2x2(nthreads, work1, max1, work2, max2, &nt1, &nt2);
    return std::make_pair(nt1, nt2);
}

}

#endif
