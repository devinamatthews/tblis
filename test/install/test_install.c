#include "tblis/tblis.h"

int main(int argc, char** argv)
{
    double data_A[10*9*2*5];
    tblis_tensor A;
    tblis_init_tensor_d(&A, 4, (len_type[]){10, 9, 2, 5},
                        data_A, (stride_type[]){1, 10, 90, 180});

    double data_B[7*5*9*8];
    tblis_tensor B;
    tblis_init_tensor_d(&B, 4, (len_type[]){7, 5, 9, 8},
                        data_B, (stride_type[]){1, 7, 35, 315});

    double data_C[7*2*10*8];
    tblis_tensor C;
    tblis_init_tensor_d(&C, 4, (len_type[]){7, 2, 10, 8},
                        data_C, (stride_type[]){1, 7, 14, 140});

    tblis_tensor_mult(NULL, NULL, &A, "cebf", &B, "afed", &C, "abcd");

    return 0;
}
