// Apple's local label requirements.
#if defined(__APPLE__)
#define LABEL(str) "   L" #str": \n\t"
#define BEQ(str) "b.eq L" #str"  \n\t"
#define BNE(str) "b.ne L" #str"  \n\t"
#define BRANCH(str) "b L" #str"  \n\t"
#else
#define LABEL(str) "   ." #str": \n\t"
#define BEQ(str) "b.eq ." #str"  \n\t"
#define BNE(str) "b.ne ." #str"  \n\t"
#define BRANCH(str) "b ." #str"  \n\t"
#endif

