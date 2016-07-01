#include "tblis.hpp"
#include "impl/tensor_impl.hpp"

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
                           T  beta,             tensor_view<T>& C, const std::string& idx_C)
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

    matrix_view<T> am, bm, cm;

    matricize(ar, am, 0);
    matricize(br, bm, 0);
    matricize(cr, cm, idx_AC.size());

    normalize(cr, idx_AC_BC);

    tensor_transpose_impl<T>(1.0, A, idx_A, 0.0, ar, idx_A);
    tensor_transpose_impl<T>(1.0, B, idx_B, 0.0, br, idx_B);
    tblis_zerov(cm.length(0)*cm.length(1), cm.data(), 1);
    ger(cm.length(0), cm.length(1),
        alpha, am.data(), 1, bm.data(), 1,
               cm.data(), cm.stride(1));
    tensor_transpose_impl<T>(1.0, cr, idx_AC_BC, beta, C, idx_C);

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_outer_prod_blas<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                       const const_tensor_view<T>& B, const std::string& idx_B, \
                              T  beta,             tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
