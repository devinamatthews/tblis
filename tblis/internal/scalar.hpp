#ifndef TBLIS_INTERNAL_SCALAR_HPP
#define TBLIS_INTERNAL_SCALAR_HPP

#include <tblis/internal/types.hpp>

namespace tblis
{
namespace internal
{

void add(type_t type, const scalar& alpha, bool conj_A, char* A,
                      const scalar&  beta, bool conj_B, char* B);

void reduce(type_t type, reduce_t op,
            char* A, len_type  idx_A,
            char* B, len_type& idx_B);

void mult(type_t type, const scalar& alpha, bool conj_A, char* A,
                                            bool conj_B, char* B,
                       const scalar&  beta, bool conj_C, char* C);

}
}

#endif //TBLIS_INTERNAL_SCALAR_HPP
