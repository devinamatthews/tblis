#ifndef _STL_EXT_ANY_HPP_
#define _STL_EXT_ANY_HPP_

#include <memory>
#include <typeinfo>

#include "type_traits.hpp"

namespace stl_ext
{

namespace detail
{
    struct any_container_base
    {
        const std::type_info& type;

        any_container_base(const std::type_info& type)
        : type(type) {}

        any_container_base(const any_container_base&) = delete;

        any_container_base(any_container_base&&) = delete;

        any_container_base& operator=(const any_container_base&) = delete;

        any_container_base& operator=(any_container_base&&) = delete;

        virtual ~any_container_base() {}

        virtual std::unique_ptr<any_container_base> clone() const = 0;
    };

    template <typename T, typename=void>
    struct any_container;

    template <typename T>
    struct any_container<T, enable_if_t<std::is_scalar<T>::value>> : any_container_base
    {
        T data;

        any_container(T obj)
        : any_container_base(typeid(T)), data(obj) {}

        std::unique_ptr<any_container_base> clone() const
        {
            return std::unique_ptr<any_container_base>(
                new any_container(data));
        }
    };

    template <typename T>
    struct any_container<T, enable_if_t<!std::is_scalar<T>::value>> : T, any_container_base
    {
        any_container(const T& obj)
        : any_container_base(typeid(T)), T(obj) {}

        any_container(T&& obj)
        : any_container_base(typeid(T)), T(std::move(obj)) {}

        std::unique_ptr<any_container_base> clone() const
        {
            return std::unique_ptr<any_container_base>(
                new any_container(static_cast<const T&>(*this)));
        }
    };
}

class any
{
    public:
        any() {}

        any(const any& other)
        : data_(other.data_->clone()) {}

        any(any&& other) = default;

        template <typename T, typename=enable_if_t<!is_same<decay_t<T>,any>::value>>
        any(T&& value)
        : data_(new detail::any_container<decay_t<T>>(std::forward<T>(value))) {}

        any& operator=(any other)
        {
            other.swap(*this);
            return *this;
        }

        template <typename T, typename=enable_if_t<!is_same<decay_t<T>,any>::value>>
        any& operator=(T&& value)
        {
            any(std::forward<T>(value)).swap(*this);
            return *this;
        }

        void clear()
        {
            data_.reset();
        }

        void swap(any& other)
        {
            using std::swap;
            swap(data_, other.data_);
        }

        bool empty() const
        {
            return !data_;
        }

        const std::type_info& type() const
        {
            return (data_ ? data_->type : typeid(void));
        }

        template <typename T>
        enable_if_t<std::is_scalar<T>::value,T&> get()
        {
            if (typeid(T) != type() || !data_) throw std::bad_cast();
            return static_cast<detail::any_container<T>&>(*data_).data;
        }

        template <typename T>
        enable_if_t<!std::is_scalar<T>::value,T&> get()
        {
            if (!data_) throw std::bad_cast();
            return dynamic_cast<T&>(*data_);
        }

        template <typename T>
        const T& get() const
        {
            return const_cast<any&>(*this).get<T>();
        }

        friend void swap(any& a, any& b)
        {
            a.swap(b);
        }

    private:
        std::unique_ptr<detail::any_container_base> data_;
};

template <typename T, typename... Args>
any make_any(Args&&... args)
{
    return any(T(std::forward<Args>(args)...));
}

}

#endif
