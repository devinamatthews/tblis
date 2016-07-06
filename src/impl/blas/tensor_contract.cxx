#include "tblis.hpp"

#include "external/lawrap/blas.h"

using namespace std;
using namespace stl_ext;
using namespace LAWrap;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_contract_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                  const const_tensor_view<T>& B, const std::string& idx_B,
                         T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    string idx_AB = intersection(idx_A, idx_B);
    string idx_AC = intersection(idx_A, idx_C);
    string idx_BC = intersection(idx_B, idx_C);

    string idx_AB_AC = idx_AB + idx_AC;
    string idx_AB_BC = idx_AB + idx_BC;
    string idx_AC_BC = idx_AC + idx_BC;

    vector<idx_type> len_AB_AC(idx_AB_AC.size());
    vector<idx_type> len_AB_BC(idx_AB_BC.size());
    vector<idx_type> len_AC_BC(idx_AC_BC.size());

    for (unsigned i = 0;i < idx_AB_AC.size();i++)
        for (unsigned j = 0;j < A.dimension();j++)
            if (idx_AB_AC[i] == idx_A[j]) len_AB_AC[i] = A.length(j);

    for (unsigned i = 0;i < idx_AB_BC.size();i++)
        for (unsigned j = 0;j < B.dimension();j++)
            if (idx_AB_BC[i] == idx_B[j]) len_AB_BC[i] = B.length(j);

    for (unsigned i = 0;i < idx_AC_BC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_AC_BC[i] == idx_C[j]) len_AC_BC[i] = C.length(j);

    tensor<T> ar(len_AB_AC);
    tensor<T> br(len_AB_BC);
    tensor<T> cr(len_AC_BC);
          tensor_view<T> arv(ar);
          tensor_view<T> brv(br);
    const_tensor_view<T> crv(cr);

    matrix_view<T> am, bm, cm;

    matricize<T>(ar, am, idx_AB.size());
    matricize<T>(br, bm, idx_AB.size());
    matricize<T>(cr, cm, idx_AC.size());

    tensor_transpose_impl<T>(1.0, A, idx_A, 0.0, arv, idx_AB_AC);
    tensor_transpose_impl<T>(1.0, B, idx_B, 0.0, brv, idx_AB_BC);
    gemm('T', 'N', cm.length(0), cm.length(1), am.length(0),
         alpha, am.data(), am.stride(1),
                bm.data(), bm.stride(1),
           0.0, cm.data(), cm.stride(1));
    tensor_transpose_impl<T>(1.0, crv, idx_AC_BC, beta, C, idx_C);

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_contract_blas<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                     const const_tensor_view<T>& B, const std::string& idx_B, \
                            T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
