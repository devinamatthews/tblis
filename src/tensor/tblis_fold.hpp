#ifndef _TBLIS_PARTITION_HPP_
#define _TBLIS_PARTITION_HPP_

#include "external/stl_ext/include/algorithm.hpp"

#include "tblis_tensor_detail.hpp"

namespace tblis
{

namespace detail
{
    template <size_t N, typename... Strides>
    struct swap_strides_helper
    {
        swap_strides_helper(std::tuple<Strides&...>& strides,
                            std::tuple<Strides...>& oldstrides)
        {
            std::get<N>(strides).swap(std::get<N>(oldstrides));
            swap_strides_helper<N+1, Strides...>(strides, oldstrides);
        }
    };

    template <typename... Strides>
    struct swap_strides_helper<sizeof...(Strides), Strides...>
    {
        swap_strides_helper(std::tuple<Strides&...>& strides,
                            std::tuple<Strides...>& oldstrides) {}
    };

    template <typename... Strides>
    void swap_strides(std::tuple<Strides&...>& strides,
                      std::tuple<Strides...>& oldstrides)
    {
        swap_strides_helper<0, Strides...>(strides, oldstrides);
    }

    template <size_t N, typename... Strides>
    struct are_contiguous_helper
    {
        bool operator()(std::tuple<Strides...>& strides,
                        const std::vector<idx_type>& lengths,
                        int i, int im1)
        {
            return std::get<N>(strides)[i] == std::get<N>(strides)[im1]*lengths[im1] &&
                are_contiguous_helper<N+1, Strides...>()(strides, lengths, i, im1);
        }
    };

    template <typename... Strides>
    struct are_contiguous_helper<sizeof...(Strides), Strides...>
    {
        bool operator()(std::tuple<Strides...>& strides,
                        const std::vector<idx_type>& lengths,
                        int i, int im1)
        {
            return true;
        }
    };

    template <typename... Strides>
    bool are_contiguous(std::tuple<Strides...>& strides,
                        const std::vector<idx_type>& lengths,
                        int i, int im1)
    {
        return are_contiguous_helper<0, Strides...>()(strides, lengths, i, im1);
    }

    template <size_t N, typename... Strides>
    struct push_back_strides_helper
    {
        push_back_strides_helper(std::tuple<Strides&...>& strides,
                                 std::tuple<Strides...>& oldstrides, int i)
        {
            std::get<N>(strides).push_back(std::get<N>(oldstrides)[i]);
            push_back_strides_helper<N+1, Strides...>(strides, oldstrides, i);
        }
    };

    template <typename... Strides>
    struct push_back_strides_helper<sizeof...(Strides), Strides...>
    {
        push_back_strides_helper(std::tuple<Strides&...>& strides,
                                 std::tuple<Strides...>& oldstrides, int i) {}
    };

    template <typename... Strides>
    void push_back_strides(std::tuple<Strides&...>& strides,
                           std::tuple<Strides...>& oldstrides, int i)
    {
        push_back_strides_helper<0, Strides...>(strides, oldstrides, i);
    }

    template <size_t N, typename... Strides>
    struct are_compatible_helper
    {
        bool operator()(const std::vector<idx_type>& len_A,
                        const std::tuple<Strides...>& stride_A,
                        const std::vector<idx_type>& len_B,
                        const std::tuple<Strides&...>& stride_B)
        {
            return are_compatible(len_A, std::get<N>(stride_A),
                                  len_B, std::get<N>(stride_B)) &&
                are_compatible_helper<N+1, Strides...>()(len_A, stride_A,
                                                         len_B, stride_B);
        }
    };

    template <typename... Strides>
    struct are_compatible_helper<sizeof...(Strides), Strides...>
    {
        bool operator()(const std::vector<idx_type>& len_A,
                        const std::tuple<Strides...>& stride_A,
                        const std::vector<idx_type>& len_B,
                        const std::tuple<Strides&...>& stride_B)
        {
            return true;
        }
    };

    template <typename... Strides>
    bool are_compatible(const std::vector<idx_type>& len_A,
                        const std::tuple<Strides...>& stride_A,
                        const std::vector<idx_type>& len_B,
                        const std::tuple<Strides&...>& stride_B)
    {
        return are_compatible_helper<0, Strides...>()(len_A, stride_A,
                                                      len_B, stride_B);
    }
}

template <typename... Strides>
void fold(std::vector<idx_type>& lengths, std::string& idx,
          Strides&... _strides)
{
    std::tuple<Strides&...> strides(_strides...);

    unsigned ndim = lengths.size();
    std::vector<unsigned> inds = range(ndim);
    stl_ext::sort(inds, detail::sort_by_stride(std::get<0>(strides)));

    std::string oldidx;
    std::vector<idx_type> oldlengths;
    std::tuple<Strides...> oldstrides;

    oldidx.swap(idx);
    oldlengths.swap(lengths);
    detail::swap_strides(strides, oldstrides);

    for (unsigned i = 0;i < ndim;i++)
    {
        if (i != 0 && detail::are_contiguous(oldstrides, oldlengths, inds[i], inds[i-1]))
        {
            lengths.back() *= oldlengths[inds[i]];
        }
        else
        {
            idx.push_back(oldidx[inds[i]]);
            lengths.push_back(oldlengths[inds[i]]);
            detail::push_back_strides(strides, oldstrides, inds[i]);
        }
    }

    for (size_t i = 0;i < strides.size();i++)
    {
        TBLIS_ASSERT(detail::are_compatible(oldlengths, oldstrides,
                                            lengths, strides));
    }
}

template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF)
{
    TBLIS_ASSERT(AF.dimension() == idx_AF.size());

    auto len_AF = AF.lengths();
    auto stride_AF = AF.strides();

    fold(len_AF, idx_AF, stride_AF);

    AF.reset(len_AF, AF.data(), stride_AF);
}

template <typename T>
void fold(tensor_view<T>& AF, std::string& idx_AF)
{
    fold(reinterpret_cast<const_tensor_view<T>&>(AF), idx_AF);
}

template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF,
          const_tensor_view<T>& BF, std::string& idx_BF)
{
    TBLIS_ASSERT(AF.dimension() == idx_AF.size());
    TBLIS_ASSERT(BF.dimension() == idx_BF.size());

    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;

    auto idx_AB = intersection(idx_AF, idx_BF);
    auto len_AB = select_from(AF.lengths(), idx_AF, idx_AB);
    auto stride_A_AB = select_from(AF.strides(), idx_AF, idx_AB);
    auto stride_B_AB = select_from(BF.strides(), idx_BF, idx_AB);

    auto idx_A_only = exclusion(idx_AF, idx_AB);
    auto len_A = select_from(AF.lengths(), idx_AF, idx_A_only);
    auto stride_A_A = select_from(AF.strides(), idx_AF, idx_A_only);

    auto idx_B_only = exclusion(idx_BF, idx_AB);
    auto len_B = select_from(BF.lengths(), idx_BF, idx_B_only);
    auto stride_B_B = select_from(BF.strides(), idx_BF, idx_B_only);

    fold(len_A, idx_A_only, stride_A_A);
    fold(len_B, idx_B_only, stride_B_B);
    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);

    AF.reset(len_A+len_AB, AF.data(), stride_A_A+stride_A_AB);
    BF.reset(len_B+len_AB, BF.data(), stride_B_B+stride_B_AB);
    idx_AF = idx_A_only+idx_AB;
    idx_BF = idx_B_only+idx_AB;
}

template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF,
                tensor_view<T>& BF, std::string& idx_BF)
{
    fold(AF, idx_AF,
         reinterpret_cast<const_tensor_view<T>&>(BF), idx_BF);
}

/*
template <typename TensorA, typename TensorB, typename TensorC>
void fold(TensorA& AF, std::string& idx_AF,
          TensorB& BF, std::string& idx_BF,
          TensorC& CF, std::string& idx_CF)
*/
template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF,
          const_tensor_view<T>& BF, std::string& idx_BF,
                tensor_view<T>& CF, std::string& idx_CF)
{
    using stl_ext::intersection;
    using stl_ext::exclusion;
    using stl_ext::select_from;

    auto idx_ABC = intersection(idx_AF, idx_BF, idx_CF);
    auto len_ABC = select_from(AF.lengths(), idx_AF, idx_ABC);
    auto stride_A_ABC = select_from(AF.strides(), idx_AF, idx_ABC);
    auto stride_B_ABC = select_from(BF.strides(), idx_BF, idx_ABC);
    auto stride_C_ABC = select_from(CF.strides(), idx_CF, idx_ABC);

    auto idx_AB = exclusion(intersection(idx_AF, idx_BF), idx_ABC);
    auto len_AB = select_from(AF.lengths(), idx_AF, idx_AB);
    auto stride_A_AB = select_from(AF.strides(), idx_AF, idx_AB);
    auto stride_B_AB = select_from(BF.strides(), idx_BF, idx_AB);

    auto idx_AC = exclusion(intersection(idx_AF, idx_CF), idx_ABC);
    auto len_AC = select_from(AF.lengths(), idx_AF, idx_AC);
    auto stride_A_AC = select_from(AF.strides(), idx_AF, idx_AC);
    auto stride_C_AC = select_from(CF.strides(), idx_CF, idx_AC);

    auto idx_BC = exclusion(intersection(idx_BF, idx_CF), idx_ABC);
    auto len_BC = select_from(BF.lengths(), idx_BF, idx_BC);
    auto stride_B_BC = select_from(BF.strides(), idx_BF, idx_BC);
    auto stride_C_BC = select_from(CF.strides(), idx_CF, idx_BC);

    auto idx_A_only = exclusion(idx_AF, idx_BF, idx_CF);
    auto len_A = select_from(AF.lengths(), idx_AF, idx_A_only);
    auto stride_A_A = select_from(AF.strides(), idx_AF, idx_A_only);

    auto idx_B_only = exclusion(idx_BF, idx_AF, idx_CF);
    auto len_B = select_from(BF.lengths(), idx_BF, idx_B_only);
    auto stride_B_B = select_from(BF.strides(), idx_BF, idx_B_only);

    auto idx_C_only = exclusion(idx_CF, idx_AF, idx_BF);
    auto len_C = select_from(CF.lengths(), idx_CF, idx_C_only);
    auto stride_C_C = select_from(CF.strides(), idx_CF, idx_C_only);

    fold(  len_A, idx_A_only,                               stride_A_A);
    fold(  len_B, idx_B_only,                               stride_B_B);
    fold(  len_C, idx_C_only,                               stride_C_C);
    fold( len_AB,     idx_AB,                stride_A_AB,  stride_B_AB);
    fold( len_AC,     idx_AC,                stride_A_AC,  stride_C_AC);
    fold( len_BC,     idx_BC,                stride_B_BC,  stride_C_BC);
    fold(len_ABC,    idx_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    /*
    unsigned idx = 0;
    for (auto l : len_A) AF.length(idx++, l);
    for (auto l : len_AB) AF.length(idx++, l);
    for (auto l : len_AC) AF.length(idx++, l);
    for (auto l : len_ABC) AF.length(idx++, l);

    idx = 0;
    for (auto l : len_B) BF.length(idx++, l);
    for (auto l : len_AB) BF.length(idx++, l);
    for (auto l : len_BC) BF.length(idx++, l);
    for (auto l : len_ABC) BF.length(idx++, l);

    idx = 0;
    for (auto l : len_C) CF.length(idx++, l);
    for (auto l : len_AC) CF.length(idx++, l);
    for (auto l : len_BC) CF.length(idx++, l);
    for (auto l : len_ABC) CF.length(idx++, l);

    idx = 0;
    for (auto s : stride_A_A) AF.stride(idx++, s);
    for (auto s : stride_A_AB) AF.stride(idx++, s);
    for (auto s : stride_A_AC) AF.stride(idx++, s);
    for (auto s : stride_A_ABC) AF.stride(idx++, s);

    idx = 0;
    for (auto s : stride_B_B) BF.stride(idx++, s);
    for (auto s : stride_B_AB) BF.stride(idx++, s);
    for (auto s : stride_B_BC) BF.stride(idx++, s);
    for (auto s : stride_B_ABC) BF.stride(idx++, s);

    idx = 0;
    for (auto s : stride_C_C) CF.stride(idx++, s);
    for (auto s : stride_C_AC) CF.stride(idx++, s);
    for (auto s : stride_C_BC) CF.stride(idx++, s);
    for (auto s : stride_C_ABC) CF.stride(idx++, s);
    */

    AF.reset(len_A+len_AB+len_AC+len_ABC, AF.data(), stride_A_A+stride_A_AB+stride_A_AC+stride_A_ABC);
    BF.reset(len_B+len_AB+len_BC+len_ABC, BF.data(), stride_B_B+stride_B_AB+stride_B_BC+stride_B_ABC);
    CF.reset(len_C+len_AC+len_BC+len_ABC, CF.data(), stride_C_C+stride_C_AC+stride_C_BC+stride_C_ABC);
    idx_AF = idx_A_only+idx_AB+idx_AC+idx_ABC;
    idx_BF = idx_B_only+idx_AB+idx_BC+idx_ABC;
    idx_CF = idx_C_only+idx_AC+idx_BC+idx_ABC;
}

}

#endif
