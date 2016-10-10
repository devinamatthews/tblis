
#define INSTANTIATE_FOR_TYPE(T) \
template void GenericMicroKernel<TBLIS_CONFIG_NAME,T> \
( \
    stride_type k, const T* TBLIS_RESTRICT alpha, \
    const T* TBLIS_RESTRICT p_a, const T* TBLIS_RESTRICT p_b, \
    const T* TBLIS_RESTRICT beta, \
    T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_A> \
( \
    len_type m, len_type k, \
    const T* p_a, stride_type rs_a, stride_type cs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_B> \
( \
    len_type m, len_type k, \
    const T* p_a, stride_type rs_a, stride_type cs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_A> \
( \
    len_type m, len_type k, \
    const T* p_a, const stride_type* TBLIS_RESTRICT rscat_a, stride_type cs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_B> \
( \
    len_type m, len_type k, \
    const T* p_a, const stride_type* TBLIS_RESTRICT rscat_a, stride_type cs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_A> \
( \
    len_type m, len_type k, \
    const T* p_a, stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_B> \
( \
    len_type m, len_type k, \
    const T* p_a, stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_A> \
( \
    len_type m, len_type k, \
    const T* p_a, const stride_type* TBLIS_RESTRICT rscat_a, \
    const stride_type* TBLIS_RESTRICT cscat_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_B> \
( \
    len_type m, len_type k, \
    const T* p_a, const stride_type* TBLIS_RESTRICT rscat_a, \
    const stride_type* TBLIS_RESTRICT cscat_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_A> \
( \
    len_type m, len_type k, \
    const T* p_a, stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a, \
    const stride_type* TBLIS_RESTRICT cbs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_B> \
( \
    len_type m, len_type k, \
    const T* p_a, stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a, \
    const stride_type* TBLIS_RESTRICT cbs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_A> \
( \
    len_type m, len_type k, \
    const T* p_a, const stride_type* TBLIS_RESTRICT rscat_a, \
    const stride_type* TBLIS_RESTRICT cscat_a, \
    const stride_type* TBLIS_RESTRICT cbs_a, \
    T* p_ap \
); \
template void PackMicroPanel<TBLIS_CONFIG_NAME,T,matrix_constants::MAT_B> \
( \
    len_type m, len_type k, \
    const T* p_a, const stride_type* TBLIS_RESTRICT rscat_a, \
    const stride_type* TBLIS_RESTRICT cscat_a, \
    const stride_type* TBLIS_RESTRICT cbs_a, \
    T* p_ap \
);

#include "tblis_instantiate_for_types.hpp"

#undef TBLIS_CONFIG_NAME
