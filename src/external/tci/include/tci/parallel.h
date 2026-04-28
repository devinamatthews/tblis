#ifndef _TCI_PARALLEL_H_
#define _TCI_PARALLEL_H_

#include "tci/tci_config.h"
#include "tci/communicator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*tci_thread_func)(tci_comm*, void*);

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity);

typedef struct tci_prime_factors
{
    unsigned n;
    unsigned sqrt_n;
    unsigned f;
} tci_prime_factors;

void tci_prime_factorization(unsigned n, tci_prime_factors* factors);

unsigned tci_next_prime_factor(tci_prime_factors* factors);

void tci_partition_2x2(unsigned nthread,
                       uint64_t work1, unsigned max1,
                       uint64_t work2, unsigned max2,
                       unsigned* nt1, unsigned* nt2);

#ifdef __cplusplus
}
#endif

#endif
