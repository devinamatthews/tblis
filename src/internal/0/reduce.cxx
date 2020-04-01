#include "reduce.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(reduce_t op, T  A, len_type  idx_A,
                         T& B, len_type& idx_B)
{
    if (op == REDUCE_SUM)
    {
        B += A;
    }
    else if (op == REDUCE_SUM_ABS)
    {
        B += std::abs(A);
    }
    else if (op == REDUCE_MAX)
    {
        if (A > B)
        {
            B = A;
            idx_B = idx_A;
        }
    }
    else if (op == REDUCE_MAX_ABS)
    {
        if (std::abs(A) > std::abs(B))
        {
            B = A;
            idx_B = idx_A;
        }
    }
    else if (op == REDUCE_MIN)
    {
        if (A < B)
        {
            B = A;
            idx_B = idx_A;
        }
    }
    else if (op == REDUCE_MIN_ABS)
    {
        if (std::abs(A) < std::abs(B))
        {
            B = A;
            idx_B = idx_A;
        }
    }
    else if (op == REDUCE_NORM_2)
    {
        B += A*A;
    }
}

void reduce(type_t type, reduce_t op,
            char* A, len_type  idx_A,
            char* B, len_type& idx_B)
{
    switch (type)
    {
        case TYPE_FLOAT:
            reduce(op, *reinterpret_cast<float*>(A), idx_A,
                       *reinterpret_cast<float*>(B), idx_B);
            break;
        case TYPE_DOUBLE:
            reduce(op, *reinterpret_cast<double*>(A), idx_A,
                       *reinterpret_cast<double*>(B), idx_B);
            break;
        case TYPE_SCOMPLEX:
            reduce(op, *reinterpret_cast<scomplex*>(A), idx_A,
                       *reinterpret_cast<scomplex*>(B), idx_B);
            break;
        case TYPE_DCOMPLEX:
            reduce(op, *reinterpret_cast<dcomplex*>(A), idx_A,
                       *reinterpret_cast<dcomplex*>(B), idx_B);
            break;
    }
}

}
}
