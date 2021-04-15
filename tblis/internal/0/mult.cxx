#include <tblis/internal/scalar.hpp>

namespace tblis
{
namespace internal
{

void mult(type_t type, const scalar& alpha, bool conj_A, char* A,
                                            bool conj_B, char* B,
                       const scalar&  beta, bool conj_C, char* C)
{
    if (beta.is_zero())
    {
        switch (type)
        {
            case FLOAT:
                *reinterpret_cast<float*>(C) =
                    alpha.data.s*(*reinterpret_cast<float*>(A)) *
                                 (*reinterpret_cast<float*>(B));
                break;
            case DOUBLE:
                *reinterpret_cast<double*>(C) =
                    alpha.data.d*(*reinterpret_cast<double*>(A)) *
                                 (*reinterpret_cast<double*>(B));
                break;
            case SCOMPLEX:
                *reinterpret_cast<scomplex*>(C) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<scomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<scomplex*>(B));
                break;
            case DCOMPLEX:
                *reinterpret_cast<dcomplex*>(C) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<dcomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<dcomplex*>(B));
                break;
        }
    }
    else
    {
        switch (type)
        {
            case FLOAT:
                *reinterpret_cast<float*>(C) =
                    alpha.data.s*(*reinterpret_cast<float*>(A)) *
                                 (*reinterpret_cast<float*>(B)) +
                     beta.data.s*(*reinterpret_cast<float*>(C));
                break;
            case DOUBLE:
                *reinterpret_cast<double*>(C) =
                    alpha.data.d*(*reinterpret_cast<double*>(A)) *
                                 (*reinterpret_cast<double*>(B)) +
                     beta.data.d*(*reinterpret_cast<double*>(C));
                break;
            case SCOMPLEX:
                *reinterpret_cast<scomplex*>(C) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<scomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<scomplex*>(B)) +
                     beta.data.c*conj(conj_C, *reinterpret_cast<scomplex*>(C));
                break;
            case DCOMPLEX:
                *reinterpret_cast<dcomplex*>(C) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<dcomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<dcomplex*>(B)) +
                     beta.data.z*conj(conj_C, *reinterpret_cast<dcomplex*>(C));
                break;
        }
    }
}

}
}
