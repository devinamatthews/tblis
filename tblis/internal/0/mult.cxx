#include "mult.hpp"

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
            case TYPE_FLOAT:
                *reinterpret_cast<float*>(C) =
                    alpha.data.s*(*reinterpret_cast<float*>(A)) *
                                 (*reinterpret_cast<float*>(B));
                break;
            case TYPE_DOUBLE:
                *reinterpret_cast<double*>(C) =
                    alpha.data.d*(*reinterpret_cast<double*>(A)) *
                                 (*reinterpret_cast<double*>(B));
                break;
            case TYPE_SCOMPLEX:
                *reinterpret_cast<scomplex*>(C) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<scomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<scomplex*>(B));
                break;
            case TYPE_DCOMPLEX:
                *reinterpret_cast<dcomplex*>(C) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<dcomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<scomplex*>(B));
                break;
        }
    }
    else
    {
        switch (type)
        {
            case TYPE_FLOAT:
                *reinterpret_cast<float*>(C) =
                    alpha.data.s*(*reinterpret_cast<float*>(A)) *
                                 (*reinterpret_cast<float*>(B)) +
                     beta.data.s*(*reinterpret_cast<float*>(C));
                break;
            case TYPE_DOUBLE:
                *reinterpret_cast<double*>(C) =
                    alpha.data.d*(*reinterpret_cast<double*>(A)) *
                                 (*reinterpret_cast<double*>(B)) +
                     beta.data.d*(*reinterpret_cast<double*>(C));
                break;
            case TYPE_SCOMPLEX:
                *reinterpret_cast<scomplex*>(C) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<scomplex*>(A)) *
                                 conj(conj_B, *reinterpret_cast<scomplex*>(B)) +
                     beta.data.c*conj(conj_C, *reinterpret_cast<scomplex*>(C));
                break;
            case TYPE_DCOMPLEX:
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
