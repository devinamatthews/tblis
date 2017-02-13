#ifndef _TCI_PARALLEL_H_
#define _TCI_PARALLEL_H_

#include "tci_config.h"

#include "communicator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*tci_thread_func)(tci_comm*, void*);

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity);

typedef struct
{
    unsigned n;
    unsigned sqrt_n;
    unsigned f;
} tci_prime_factors;

void tci_prime_factorization(unsigned n, tci_prime_factors* factors);

unsigned tci_next_prime_factor(tci_prime_factors* factors);

void tci_partition_2x2(unsigned nthread, uint64_t work1, uint64_t work2,
                       unsigned* nt1, unsigned* nt2);

#ifdef __cplusplus
}

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
            Body& body = *static_cast<Body*>(data);
            body(*reinterpret_cast<communicator*>(comm));
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
    tci_partition_2x2(nthreads, work1, work2, &nt1, &nt2);
    return std::make_pair(nt1, nt2);
}

}

#endif

#endif
