#pragma once

#include "mcp_messages.h"
#include <string>
#include <tuple>
#include <vector>

namespace toolcall
{
    template <typename T>
    using callback = std::function<void(const T&)>;

    template <typename Derived, typename... MessageTypes>
    class mcp_transport_t {
    public:

        template <typename T>
        void subscribe(callback<T> callback) {
            auto& vec = std::get<std::vector<toolcall::callback<T>>>(subscribers_);
            vec.push_back(std::move(callback));
        }

        template <typename T>
        void notify(const T & message) const {
            const auto& vec = std::get<std::vector<toolcall::callback<T>>>(subscribers_);
            for (const auto& callback : vec) {
                callback(message);
            }
        }

        template <typename T>
        void notify_if(const mcp::message_variant & message) {
            if (std::holds_alternative<T>(message)) {
                notify(std::get<T>(message));
            }
        }

        template <typename T>
        bool send(const T & message) {
            return static_cast<Derived*>(this)->send(message.toJson());
        }

    private:
        std::tuple<std::vector<toolcall::callback<MessageTypes>>...> subscribers_;
    };

    class mcp_transport : public mcp_transport_t <mcp_transport,
                                                   mcp::initialize_request,
                                                   mcp::initialize_response,
                                                   mcp::initialized_notification,
                                                   mcp::tools_list_request,
                                                   mcp::tools_list_response>
    {
    public:
        virtual ~mcp_transport() = default;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual bool send(const std::string & request_json) = 0;
    };
}
