//#define PREFETCH_C_L2

#define A_L1_PREFETCH_DIST 4 //should be multiple of 2

/*The pointer of B is moved ahead by one iteration of k
before the loop starts.Therefore, prefetching 3 k iterations
ahead*/
#define B_L1_PREFETCH_DIST 4

#define TAIL_NITER 8

//#define PREFETCH_A_BEFORE
//#define PREFETCH_B_BEFORE
//#define PREFETCH_A_AFTER
//#define PREFETCH_B_AFTER
