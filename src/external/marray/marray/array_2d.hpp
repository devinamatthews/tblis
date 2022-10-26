#ifndef MARRAY_ARRAY_2D_HPP
#define MARRAY_ARRAY_2D_HPP

#include "array_1d.hpp"

namespace MArray
{
namespace detail
{

template <typename T>
struct is_matrix : std::false_type {};

template <typename T, typename D, bool O>
struct is_matrix<marray_base<T, 2, D, O>> : std::true_type {};

template <typename T>
struct is_matrix<marray_view<T, 2>> : std::true_type {};

template <typename T, typename A>
struct is_matrix<marray<T, 2, A>> : std::true_type {};

template <typename T, typename U, typename=void>
struct is_matrix_of : std::false_type {};

template <typename T, typename U>
struct is_matrix_of<T, U, std::enable_if_t<is_matrix<T>::value &&
    std::is_assignable<U&,typename T::value_type>::value>> : std::true_type {};

template <typename T, typename U, typename V=void>
using enable_if_matrix_of_t = std::enable_if_t<is_matrix_of<T,U>::value, V>;

template <typename T, typename U>
struct is_2d_container_of :
    std::integral_constant<bool, is_matrix_of<T,U>::value ||
                                 is_container_of_containers_of<T,U>::value> {};

template <typename T, typename U, typename V=void>
using enable_if_2d_container_of_t = std::enable_if_t<is_2d_container_of<T,U>::value, V>;

template <typename T>
std::enable_if_t<is_container_of_containers<T>::value,len_type>
length(const T& len, int dim)
{
    if (dim == 0) return len.size();
    else
    {
        MARRAY_ASSERT(dim == 1);
        auto it = len.begin();
        if (it == len.end()) return 0;
        len_type l = it->size();
        while (++it != len.end()) MARRAY_ASSERT((len_type)it->size() == l);
        return l;
    }
}

template <typename T>
std::enable_if_t<is_matrix<T>::value,len_type>
length(const T& len, int dim)
{
    return len.length(dim);
}

}

/**
 * Adaptor class which can capture a 2-d data structure: a container of containers, @ref matrix or @ref matrix_view,
 * nested initializer list, etc. which have elements convertible to a specified type.
 *
 * This class is used to capture a variety of acceptable argument types without relying on template parameters.
 * For example:
 *
 * @code{.cxx}
 * void foo(const array_2d<int>&);
 *
 * // An initializer_list<initializer_list<int>> (sort of...actually narrowing conversions are allowed):
 * foo({{1, 4}, {4l, 3u}});
 *
 * // A container of containers:
 * std::vector<std::list<long>> vv = {{1, 5}, {2, 1}};
 * foo(vv);
 *
 * // An initializer list of containers:
 * std::vector<int> v1{1, 6};
 * std::vector<int> v2{2, 0};
 * foo({v1, v2});
 *
 * // A matrix (2-d tensor):
 * matrix<int> m = {{3, 4}, {0, 2}};
 * foo(m);
 *
 * // A matrix_view:
 * std::vector<int> v{6, 3, 0, 1};
 * matrix_view<int> mv({2, 2}, v.data(), ROW_MAJOR); // view v as a 2-d structure: {{6, 3}, {0, 1}}
 * foo(mv);
 * @endcode
 *
 * @tparam T  Elements of the supplied container, initializer list, etc. must be convertible to this type.
 *
 * @ingroup util
 */
template <typename T>
class array_2d
{
    protected:
        struct adaptor_base
        {
            std::array<len_type,2> len;

            adaptor_base(len_type len0, len_type len1) : len{len0, len1} {}

            virtual ~adaptor_base() {}

            virtual void slurp(T* x, len_type rs, len_type cs) const = 0;

            virtual void slurp(std::vector<std::vector<T>>&) const = 0;
        };

        template <typename U>
        struct adaptor : adaptor_base
        {
            static constexpr bool IsMatrix = detail::is_matrix<typename std::decay<U>::type>::value;

            U data;
            using adaptor_base::len;

            adaptor(U data)
            : adaptor_base(detail::length(data, 0), detail::length(data, 1)),
              data(data) {}

            template <bool IsMatrix_ = IsMatrix>
            std::enable_if_t<!IsMatrix_>
            do_slurp(T* x, len_type rs, len_type cs) const
            {
                int i = 0;
                for (auto it = data.begin(), end = data.end();it != end;++it)
                {
                    int j = 0;
                    for (auto it2 = it->begin(), end2 = it->end();it2 != end2;++it2)
                    {
                        x[i*rs + j*cs] = *it2;
                        j++;
                    }
                    i++;
                }
            }

            template <bool IsMatrix_ = IsMatrix>
            std::enable_if_t<!IsMatrix_>
            do_slurp(std::vector<std::vector<T>>& x) const
            {
                x.clear();
                for (auto it = data.begin(), end = data.end();it != end;++it)
                {
                    x.emplace_back(it->begin(), it->end());
                }
            }

            template <bool IsMatrix_ = IsMatrix>
            std::enable_if_t<IsMatrix_>
            do_slurp(T* x, len_type rs, len_type cs) const
            {
                for (len_type i = 0;i < len[0];i++)
                {
                    for (len_type j = 0;j < len[1];j++)
                    {
                        x[i*rs + j*cs] = data[i][j];
                    }
                }
            }

            template <bool IsMatrix_ = IsMatrix>
            std::enable_if_t<IsMatrix_>
            do_slurp(std::vector<std::vector<T>>& x) const
            {
                x.resize(len[0]);
                for (len_type i = 0;i < len[0];i++)
                {
                    x[i].resize(len[1]);
                    for (len_type j = 0;j < len[1];j++)
                    {
                        x[i][j] = data[i][j];
                    }
                }
            }

            virtual void slurp(T* x, len_type rs, len_type cs) const override
            {
                do_slurp(x, rs, cs);
            }

            virtual void slurp(std::vector<std::vector<T>>& x) const override
            {
                do_slurp(x);
            }
        };

        static constexpr size_t _s1 = sizeof(adaptor<std::initializer_list<std::initializer_list<int>>>);
        static constexpr size_t _s2 = sizeof(adaptor<const matrix<int>&>);
        static constexpr size_t max_adaptor_size = (_s1 > _s2 ? _s1 : _s2);

        std::aligned_storage_t<max_adaptor_size> raw_adaptor_;
        adaptor_base& adaptor_;

        template <typename U>
        adaptor_base& adapt(U data)
        {
            return *(new (&raw_adaptor_) adaptor<U>(data));
        }

        adaptor_base& adapt(const array_2d& other)
        {
            memcpy(&raw_adaptor_, &other.raw_adaptor_, sizeof(raw_adaptor_));

            return *reinterpret_cast<adaptor_base*>(
                reinterpret_cast<char*>(this) +
                (reinterpret_cast<const char*>(&other.adaptor_) -
                 reinterpret_cast<const char*>(&other)));
        }

    public:
        array_2d(const array_2d& other)
        : adaptor_(adapt(other)) {}

        array_2d(std::initializer_list<std::initializer_list<T>> data)
        : adaptor_(adapt<std::initializer_list<std::initializer_list<T>>>(data)) {}

        template <typename U, typename=std::enable_if_t<std::is_assignable_v<T&,U>>>
        array_2d(std::initializer_list<std::initializer_list<U>> data)
        : adaptor_(adapt<std::initializer_list<std::initializer_list<U>>>(data)) {}

        template <typename U, typename=std::enable_if_t<detail::is_1d_container_of<U,T>::value>>
        array_2d(std::initializer_list<U> data)
        : adaptor_(adapt<std::initializer_list<U>>(data)) {}

        template <typename U, typename=std::enable_if_t<detail::is_2d_container_of<U,T>::value>>
        array_2d(const U& data)
        : adaptor_(adapt<const U&>(data)) {}

        ~array_2d() { adaptor_.~adaptor_base(); }

        void slurp(std::vector<std::vector<T>>& x) const { adaptor_.slurp(x); }

        template <size_t M, size_t N>
        void slurp(std::array<std::array<T, N>, M>& x) const
        {
            MARRAY_ASSERT((len_type)M >= length(0));
            MARRAY_ASSERT((len_type)N >= length(1));
            adaptor_.slurp(&x[0][0], N, 1);
        }

        template <size_t M, size_t N>
        void slurp(short_vector<std::array<T, N>, M>& x) const
        {
            x.resize(length(0));
            MARRAY_ASSERT((len_type)N >= length(1));
            adaptor_.slurp(&x[0][0], N, 1);
        }

        void slurp(matrix<T>& x, layout layout = DEFAULT_LAYOUT) const
        {
            x.reset({length(0), length(1)}, layout);
            adaptor_.slurp(x.data(), x.stride(0), x.stride(1));
        }

        len_type length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < 2);
            return adaptor_.len[dim];
        }
};

}

#endif //MARRAY_ARRAY_2D_HPP
