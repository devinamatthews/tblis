#include <tblis/internal/scalar.hpp>

namespace tblis
{
namespace internal
{

void add(type_t type, const scalar& alpha, bool conj_A, char* A,
                      const scalar&  beta, bool conj_B, char* B)
{
    if (beta.is_zero())
    {
        switch (type)
        {
            case FLOAT:
                *reinterpret_cast<float*>(B) =
                    alpha.data.s*(*reinterpret_cast<float*>(A));
                break;
            case DOUBLE:
                *reinterpret_cast<double*>(B) =
                    alpha.data.d*(*reinterpret_cast<double*>(A));
                break;
            case SCOMPLEX:
                *reinterpret_cast<scomplex*>(B) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<scomplex*>(A));
                break;
            case DCOMPLEX:
                *reinterpret_cast<dcomplex*>(B) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<dcomplex*>(A));
                break;
        }
    }
    else
    {
        switch (type)
        {
            case FLOAT:
                *reinterpret_cast<float*>(B) =
                    alpha.data.s*(*reinterpret_cast<float*>(A)) +
                    beta.data.s*(*reinterpret_cast<float*>(B));
                break;
            case DOUBLE:
                *reinterpret_cast<double*>(B) =
                    alpha.data.d*(*reinterpret_cast<double*>(A)) +
                    beta.data.d*(*reinterpret_cast<double*>(B));
                break;
            case SCOMPLEX:
                *reinterpret_cast<scomplex*>(B) =
                    alpha.data.c*conj(conj_A, *reinterpret_cast<scomplex*>(A)) +
                    beta.data.c*conj(conj_B, *reinterpret_cast<scomplex*>(B));
                break;
            case DCOMPLEX:
                *reinterpret_cast<dcomplex*>(B) =
                    alpha.data.z*conj(conj_A, *reinterpret_cast<dcomplex*>(A)) +
                    beta.data.z*conj(conj_B, *reinterpret_cast<dcomplex*>(B));
                break;
        }
    }
}

}
}
