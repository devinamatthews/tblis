#include "tci/parallel.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>

#if TCI_USE_OPENMP_THREADS

#include <omp.h>

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity)
{
    if (nthread <= 1)
    {
        func(tci_single, payload);
        return 0;
    }

    tci_context* context;
    int ret = tci_context_init(&context, nthread, arity);
    if (ret != 0) return ret;

    #pragma omp parallel num_threads(nthread)
    {
        tci_comm comm;

        if (omp_get_num_threads() != nthread)
        {
            tci_comm_init_single(&comm);
        }
        else
        {
            tci_comm_init(&comm, context,
                          nthread, (unsigned)omp_get_thread_num(), 1, 0);
        }

        func(&comm, payload);
        #pragma omp barrier

        tci_comm_destroy(&comm);
    }

    return 0;
}

#elif TCI_USE_OMPTASK_THREADS

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity)
{
    #pragma omp parallel num_threads(nthread)
    {
        #pragma omp single
        func(tci_single, payload);
    }

    return 0;
}

#elif TCI_USE_PTHREADS_THREADS

#include <pthread.h>

typedef struct
{
    tci_thread_func func;
    void* payload;
    tci_context* context;
    unsigned nthread, tid;
} tci_thread_data;

void* tci_run_thread(void* raw_data)
{
    tci_thread_data* data = (tci_thread_data*)raw_data;
    tci_comm comm;
    tci_comm_init(&comm, data->context, data->nthread, data->tid, 1, 0);
    data->func(&comm, data->payload);
    tci_comm_destroy(&comm);
    return NULL;
}

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity)
{
    if (nthread <= 1)
    {
        func(tci_single, payload);
        return 0;
    }

    tci_context* context;
    int ret = tci_context_init(&context, nthread, arity);
    if (ret != 0) return ret;

    pthread_t threads[nthread];
    tci_thread_data data[nthread];

    tci_comm comm0;
    tci_comm_init(&comm0, context, nthread, 0, 1, 0);

    for (unsigned i = 1;i < nthread;i++)
    {
        data[i].func = func;
        data[i].payload = payload;
        data[i].context = context;
        data[i].nthread = nthread;
        data[i].tid = i;

        int ret = pthread_create(&threads[i], NULL, tci_run_thread, &data[i]);
        if (ret != 0)
        {
            for (unsigned j = 1;j < i;j++) pthread_join(threads[j], NULL);
            return ret;
        }
    }

    func(&comm0, payload);

    for (unsigned i = 1;i < nthread;i++)
    {
        pthread_join(threads[i], NULL);
    }

    return tci_comm_destroy(&comm0);
}

#elif TCI_USE_WINDOWS_THREADS

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef struct
{
    tci_thread_func func;
    void* payload;
    tci_context* context;
    unsigned nthread, tid;
} tci_thread_data;

DWORD WINAPI tci_run_thread(void* raw_data)
{
    tci_thread_data* data = (tci_thread_data*)raw_data;
    tci_comm comm;
    tci_comm_init(&comm, data->context, data->nthread, data->tid, 1, 0);
    data->func(&comm, data->payload);
    tci_comm_destroy(&comm);
    return NULL;
}

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity)
{
    if (nthread <= 1)
    {
        func(tci_single, payload);
        return 0;
    }

    tci_context* context;
    int ret = tci_context_init(&context, nthread, arity);
    if (ret != 0) return ret;

    HANDLE threads[nthread-1];
    tci_thread_data data[nthread-1];

    for (unsigned i = 0;i < nthread-1;i++)
    {
        data[i].func = func;
        data[i].payload = payload;
        data[i].context = context;
        data[i].nthread = nthread;
        data[i].tid = i+1;

        threads[i] = CreateThread(NULL, 0, tci_run_thread, &data[i], 0, NULL);
        if (!threads[i])
        {
            WaitForMultipleObjects(i, threads, TRUE, INFINITE);
            return -1;
        }
    }

    tci_comm comm0;
    tci_comm_init(&comm0, context, nthread, 0, 1, 0);
    func(&comm0, payload);

    WaitForMultipleObjects(nthread-1, threads, TRUE, INFINITE);

    return tci_comm_destroy(&comm0);
}

#elif TCI_IS_TASK_BASED

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity)
{
    tci_comm comm = {NULL, 1, 0, nthread, 0};
    func(&comm, payload);
    return 0;
}

#else //single threaded

int tci_parallelize(tci_thread_func func, void* payload,
                    unsigned nthread, unsigned arity)
{
    func(tci_single, payload);
    return 0;
}

#endif

void tci_prime_factorization(unsigned n, tci_prime_factors* factors)
{
    factors->n = n;
    // all this is necessary to appease the warning gods
    factors->sqrt_n = (unsigned)lrint(floor(sqrt(n)));
    factors->f = 2;
}

unsigned tci_next_prime_factor(tci_prime_factors* factors)
{
    for (;factors->f <= factors->sqrt_n;)
    {
        if (factors->f == 2)
        {
            if (factors->n%2 == 0)
            {
                factors->n /= 2;
                return 2;
            }
            factors->f = 3;
        }
        else if (factors->f == 3)
        {
            if (factors->n%3 == 0)
            {
                factors->n /= 3;
                return 3;
            }
            factors->f = 5;
        }
        else if (factors->f == 5)
        {
            if (factors->n%5 == 0)
            {
                factors->n /= 5;
                return 5;
            }
            factors->f = 7;
        }
        else if (factors->f == 7)
        {
            if (factors->n%7 == 0)
            {
                factors->n /= 7;
                return 7;
            }
            factors->f = 11;
        }
        else
        {
            if (factors->n%factors->f == 0)
            {
                factors->n /= factors->f;
                return factors->f;
            }
            factors->f++;
        }
    }

    if (factors->n != 1)
    {
        unsigned tmp = factors->n;
        factors->n = 1;
        return tmp;
    }

    return 1;
}

#define TCI_USE_EXPENSIVE_PARTITION 0

#if TCI_USE_EXPENSIVE_PARTITION

/*
 * Assumes base > 0 and power >= 0.
 */
static int ipow(int base, int power)
{
    int p = 1;

    for (int mask = 0x1;mask <= power;mask <<= 1)
    {
        if (power&mask) p *= base;
        base *= base;
    }

    return p;
}

#endif

void tci_partition_2x2(unsigned nthread,
                       uint64_t work1, unsigned max1,
                       uint64_t work2, unsigned max2,
                       unsigned* nt1, unsigned* nt2)
{
    max1 = TCI_MIN(TCI_MAX(max1, 1), nthread);
    max2 = TCI_MIN(TCI_MAX(max2, 1), nthread);

    if (nthread < 4)
    {
        if (max2 < max1 || (max1 == max2 && work1 >= work2))
        {
            *nt1 = nthread;
            *nt2 = 1;
        }
        else
        {
            *nt1 = 1;
            *nt2 = nthread;
        }
        return;
    }

    tci_prime_factors factors;
    tci_prime_factorization(nthread, &factors);

    #if !TCI_USE_EXPENSIVE_PARTITION

    unsigned num1 = 1;
    unsigned num2 = 1;

    unsigned f;
    while ((f = tci_next_prime_factor(&factors)) > 1)
    {
        if ((work2 >= work1 || num1*f > max1) && num2*f <= max2)
        {
            work2 /= f;
            num2 *= f;
        }
        else
        {
            work1 /= f;
            num1 *= f;
        }
    }

    *nt1 = num1;
    *nt2 = num2;

    #else

    /*
     * Eight distinct prime factors handles all numbers up to 223092870
     */
    int fact[8];
    int mult[8];

    int nfact = 1;
    fact[0] = tci_next_prime_factor(&factors);
    mult[0] = 1;

    int f;
    while ((f = tci_next_prime_factor(&factors)) > 1)
    {
        if (f == fact[nfact-1])
        {
            mult[nfact-1]++;
        }
        else
        {
            nfact++;
            fact[nfact-1] = f;
            mult[nfact-1] = 1;
        }
    }

    int ntake[8] = {0};
    int64_t min_diff = INT64_MAX;

    bool done = false;
    while (!done)
    {
        int x = 1;
        int y = 1;

        for (int i = 0;i < nfact;i++)
        {
            x *= ipow(fact[i], ntake[i]);
            y *= ipow(fact[i], mult[i]-ntake[i]);
        }

        int64_t diff = llabs(x*work2 - y*work1);
        if (diff < min_diff)
        {
            min_diff = diff;
            *nt1 = x;
            *nt2 = y;
        }

        for (int i = 0;i < nfact;i++)
        {
            if (++ntake[i] > mult[i])
            {
                ntake[i] = 0;
                if (i == nfact-1) done = true;
                else continue;
            }
            break;
        }
    }

    #endif

    assert((*nt1)*(*nt2) == nthread);
}
