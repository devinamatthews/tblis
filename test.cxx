#include <tblis/tblis.h>

using namespace tblis;

TBLIS_EXPORT
void tblis_tensor_add(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_tensor* A,
                      const label_type* idx_A,
                            tblis_tensor* B,
                      const label_type* idx_B)
{
}

namespace tblis
{

void add(const communicator& comm,
         const scalar& alpha,
         const const_tensor& A_,
         const label_string& idx_A,
         const scalar& beta,
         const tensor& B_,
         const label_string& idx_B)
{
    tensor A(A_);
    tensor B(B_);
    A.scalar *= alpha.convert(A.type);
    B.scalar *= beta.convert(B.type);
    tblis_tensor_add(comm, nullptr, &A, idx_A.idx, &B, idx_B.idx);
}

}

int main()
{
    double* x;
    long* y;

    add({x, 3, y, y}, "abc", {x, 3, y, y}, "abc");
}

