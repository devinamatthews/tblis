#include "tblis.hpp"

#include "external/lawrap/blas.h"

using namespace std;
using namespace stl_ext;
using namespace LAWrap;
using namespace MArray;

namespace tblis
{
namespace impl
{

template <typename T>
int tensor_weight_blas(T alpha, const const_tensor_view<T>& A, const std::string& idx_A,
                                const const_tensor_view<T>& B, const std::string& idx_B,
                       T  beta, const       tensor_view<T>& C, const std::string& idx_C)
{
    string idx_ABC = intersection(idx_A, idx_B, idx_C);
    string idx_A_not_ABC = exclusion(idx_A, idx_ABC);
    string idx_B_not_ABC = exclusion(idx_B, idx_ABC);
    string idx_C_not_ABC = exclusion(idx_C, idx_ABC);

    string idx_AC = exclusion(intersection(idx_A, idx_C), idx_ABC);
    string idx_BC = exclusion(intersection(idx_B, idx_C), idx_ABC);
    string idx_AC_BC = idx_AC + idx_BC;

    vector<idx_type> len_AC_BC(idx_AC_BC.size());

    for (unsigned i = 0;i < idx_AC_BC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_AC_BC[i] == idx_C[j]) len_AC_BC[i] = C.length(j);

    vector<idx_type> len_A_not_ABC(idx_A_not_ABC.size());
    vector<idx_type> len_B_not_ABC(idx_B_not_ABC.size());
    vector<idx_type> len_C_not_ABC(idx_C_not_ABC.size());
    vector<stride_type> stride_A_not_ABC(idx_A_not_ABC.size());
    vector<stride_type> stride_B_not_ABC(idx_B_not_ABC.size());
    vector<stride_type> stride_C_not_ABC(idx_C_not_ABC.size());

    for (unsigned i = 0;i < idx_A_not_ABC.size();i++)
        for (unsigned j = 0;j < A.dimension();j++)
            if (idx_A_not_ABC[i] == idx_A[j])
            {
                len_A_not_ABC[i] = A.length(j);
                stride_A_not_ABC[i] = A.stride(j);
            }

    for (unsigned i = 0;i < idx_B_not_ABC.size();i++)
        for (unsigned j = 0;j < B.dimension();j++)
            if (idx_B_not_ABC[i] == idx_B[j])
            {
                len_B_not_ABC[i] = B.length(j);
                stride_B_not_ABC[i] = B.stride(j);
            }

    for (unsigned i = 0;i < idx_C_not_ABC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_C_not_ABC[i] == idx_C[j])
            {
                len_C_not_ABC[i] = C.length(j);
                stride_C_not_ABC[i] = C.stride(j);
            }

    vector<idx_type> len_ABC(idx_ABC.size());
    vector<stride_type> stride_A_ABC(idx_ABC.size());
    vector<stride_type> stride_B_ABC(idx_ABC.size());
    vector<stride_type> stride_C_ABC(idx_ABC.size());

    for (unsigned i = 0;i < idx_ABC.size();i++)
        for (unsigned j = 0;j < A.dimension();j++)
            if (idx_ABC[i] == idx_A[j])
            {
                len_ABC[i] = A.length(j);
                stride_A_ABC[i] = A.stride(j);
            }

    for (unsigned i = 0;i < idx_ABC.size();i++)
        for (unsigned j = 0;j < B.dimension();j++)
            if (idx_ABC[i] == idx_B[j]) stride_B_ABC[i] = B.stride(j);

    for (unsigned i = 0;i < idx_ABC.size();i++)
        for (unsigned j = 0;j < C.dimension();j++)
            if (idx_ABC[i] == idx_C[j]) stride_C_ABC[i] = C.stride(j);

    tensor<T> ar(len_A_not_ABC);
    tensor<T> br(len_B_not_ABC);
    tensor<T> cr(len_AC_BC);
          tensor_view<T> arv(ar);
          tensor_view<T> brv(br);
    const_tensor_view<T> crv(cr);

    matrix_view<T> am, bm, cm;

    matricize<T>(ar, am, 0);
    matricize<T>(br, bm, 0);
    matricize<T>(cr, cm, idx_AC.size());

    const T* ptr_A = A.data();
    const T* ptr_B = B.data();
          T* ptr_C = C.data();

    const_tensor_view<T> A_not_ABC(len_A_not_ABC, ptr_A, stride_A_not_ABC);
    const_tensor_view<T> B_not_ABC(len_B_not_ABC, ptr_B, stride_B_not_ABC);
          tensor_view<T> C_not_ABC(len_C_not_ABC, ptr_C, stride_C_not_ABC);

    viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    while (it.next(ptr_A, ptr_B, ptr_C))
    {
        A_not_ABC.data(ptr_A);
        B_not_ABC.data(ptr_B);
        C_not_ABC.data(ptr_C);

        tensor_transpose_impl<T>(1.0, A_not_ABC, idx_A_not_ABC, 0.0, arv, idx_A_not_ABC);
        tensor_transpose_impl<T>(1.0, B_not_ABC, idx_B_not_ABC, 0.0, brv, idx_B_not_ABC);
        tblis_zerov(cm.length(0)*cm.length(1), cm.data(), 1);
        ger(cm.length(0), cm.length(1),
            alpha, am.data(), 1, bm.data(), 1,
            cm.data(), cm.stride(1));
        tensor_transpose_impl<T>(1.0, crv, idx_AC_BC, beta, C_not_ABC, idx_C_not_ABC);
    }

    return 0;
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
int tensor_weight_blas<T>(T alpha, const const_tensor_view<T>& A, const std::string& idx_A, \
                                   const const_tensor_view<T>& B, const std::string& idx_B, \
                          T  beta, const       tensor_view<T>& C, const std::string& idx_C);
#include "tblis_instantiate_for_types.hpp"

}
}
