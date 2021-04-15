// Current used only to support
//  Apple's twisted label requirements.

#if defined(__APPLE__)
#define LABEL(str) " LBB_" #str": \n\t"
#define BEQ(str) "b.eq LBB_" #str" \n\t"
#define BNE(str) "b.ne LBB_" #str" \n\t"
#define BRANCH(str) "b LBB_" #str" \n\t"
#else
#define LABEL(str) " ." #str": \n\t"
#define BEQ(str) "b.eq ." #str" \n\t"
#define BNE(str) "b.ne ." #str" \n\t"
#define BRANCH(str) "b ." #str" \n\t"
#endif

