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
int tensor_outer_prod_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                    const const_tensor_view<T>& B, const std::string& idx_B,
                           T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    string idx_AC = intersection(idx_A, idx_C);
    string idx_BC = intersection(idx_B, idx_C);

    string idx_AC_BC = idx_AC + idx_BC;

    vector<idx_type> len_AC_BC(idx_AC_BC.size());

    for (unsigned i = 0;i < idx_AC_BC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_AC_BC[i] == idx_C[j]) len_AC_BC[i] = C.length(j);

    tensor<T> ar(A.lengths());
    tensor<T> br(B.lengths());
    tensor<T> cr(len_AC_BC);
          tensor_view<T> arv(ar);
          tensor_view<T> brv(br);
    const_tensor_view<T> crv(cr);

    matrix_view<T> am, bm, cm;

    matricize<T>(ar, am, 0);
    matricize<T>(br, bm, 0);
    matricize<T>(cr, cm, idx_AC.size());

    tensor_transpose_impl<T>(1.0, A, idx_A, 0.0, arv, idx_A);
    tensor_transpose_impl<T>(1.0, B, idx_B, 0.0, brv, idx_B);
    tblis_zerov(cm.length(0)*cm.length(1), cm.data(), 1);
    ger(cm.length(0), cm.length(1),
        alpha, am.data(), 1, bm.data(), 1,
               cm.data(), cm.stride(1));
    tensor_transpose_impl<T>(1.0, crv, idx_AC_BC, beta, C, idx_C);

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_outer_prod_blas<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                       const const_tensor_view<T>& B, const std::string& idx_B, \
                              T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
