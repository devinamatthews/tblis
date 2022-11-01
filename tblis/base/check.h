#ifndef TBLIS_BASE_CHECK_H
#define TBLIS_BASE_CHECK_H

#include <tblis/base/types.h>

TBLIS_BEGIN_NAMESPACE

TBLIS_DEFINE_ENUM
    (SUCCESS, 0)
    (INVALID_DIMENSION, 1)
    (INVALID_LENGTH, 1)
    (INDEX_MISMATCH, 2)
(check_t)

TBLIS_EXPORT tblis_check_t tblis_check_tensor(int ndim,
                                              tblis_len_type* len,
                                              tblis_stride_type* stride,
                                              tblis_label_type* idx)
{
    if (ndim <= 0)
        return TBLIS_INVALID_DIMENSION;

    for (int i = 0;i < ndim;i++)
    {
        if (len[i] < 0)
            return TBLIS_INVALID_LENGTH;

        for (int j = i+1;j < ndim;j++)
            if (idx[i] == idx[j] && len[i] != len[j])
                return TBLIS_INDEX_MISMATCH;
    }

    return TBLIS_SUCCESS;
}

TBLIS_EXPORT tblis_check_t tblis_check_tensor_pair(int ndim1,
                                                   tblis_len_type* len1,
                                                   tblis_stride_type* stride1,
                                                   tblis_label_type* idx1,
                                                   int ndim2,
                                                   tblis_len_type* len2,
                                                   tblis_stride_type* stride2,
                                                   tblis_label_type* idx2)
{
    for (int i = 0;i < ndim1;i++)
    for (int j = 0;j < ndim2;j++)
        if (idx1[i] == idx2[j] && len1[i] != len2[j])
            return TBLIS_INDEX_MISMATCH;

    return TBLIS_SUCCESS;
}

TBLIS_END_NAMESPACE

#endif //TBLIS_BASE_CHECK_H
