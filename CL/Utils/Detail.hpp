#pragma once

// STL includes
#include <stddef.h>
#include <utility> // std::forward, std::integer_sequence
#include <tuple> // std::tuple, std::get
#include <initializer_list> // std::initializer_list

namespace cl {
namespace util {
    namespace detail {
        // Borrowed from:
        // https://www.fluentcpp.com/2019/03/05/for_each_arg-applying-a-function-to-each-argument-of-a-function-in-cpp/
        template <class F, class... Args> F for_each_arg(F f, Args&&... args)
        {
            (void)std::initializer_list<int>{ (
                (void)f(std::forward<Args>(args)), 0)... };
            return f;
        }

        namespace impl {
            // Borrowed from: https://stackoverflow.com/a/16387374/1476661
            template <typename T, typename F, int... Is>
            void for_each_in_tuple(T&& t, F&& f,
                                   std::integer_sequence<int, Is...>)
            {
                auto l = {
                    (std::forward<F>(f)(std::get<Is>(std::forward<T>(t))), 0)...
                };
                (void)l;
            }
        }
        template <typename... Ts, typename F>
        void for_each_in_tuple(std::tuple<Ts...> const& t, F&& f)
        {
            impl::for_each_in_tuple(
                t, std::forward<F>(f),
                std::make_integer_sequence<int, sizeof...(Ts)>());
        }

        namespace impl {
            // Borrowed from
            // https://codereview.stackexchange.com/questions/193420/apply-a-function-to-each-element-of-a-tuple-map-a-tuple
            template <class F, typename Tuple, std::size_t... Is>
            auto transform_tuple(Tuple&& t, F&& f, std::index_sequence<Is...>)
            {
                return std::make_tuple(std::forward<F>(f)(std::get<Is>(t))...);
            }
        }
        template <class F, typename... Args>
        auto transform_tuple(const std::tuple<Args...>& t, F&& f)
        {
            return impl::transform_tuple(
                t, std::forward<F>(f),
                std::make_index_sequence<sizeof...(Args)>{});
        }

        namespace impl {
            // Borrowed from
            // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3658.html
            // with modifications of Casey Carter at
            // https://stackoverflow.com/a/51365112/1476661
            template <typename F, typename Tuple, std::size_t... I>
            auto apply(F&& f, Tuple&& args, std::index_sequence<I...>)
                -> decltype(std::forward<F>(f)(
                    std::get<I>(std::forward<Tuple>(args))...))
            {
                return std::forward<F>(f)(
                    std::get<I>(std::forward<Tuple>(args))...);
            }
        }
        template <typename F, typename Tuple,
                  typename Indices = std::make_index_sequence<
                      std::tuple_size<std::remove_reference_t<Tuple>>::value>>
        auto apply(F&& f, Tuple&& args)
            -> decltype(impl::apply(std::forward<F>(f),
                                    std::forward<Tuple>(args), Indices()))
        {
            return impl::apply(std::forward<F>(f), std::forward<Tuple>(args),
                               Indices());
        }
    }
}
}
