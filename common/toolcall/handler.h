#pragma once

#include "params.h"
#include <string>
#include <variant>
#include <memory>
#include <vector>

namespace toolcall
{
    enum action {
        ACCEPT,
        PENDING,
        DEFER
    };

    class handler_impl;
    class handler {
    public:
        using ptr = std::shared_ptr<handler>;

        handler(std::unique_ptr<handler_impl> impl) : impl_(std::move(impl)) {}

        std::string tool_list();
        action call(const std::string & request, std::string & response);
        const std::string & tool_choice() const;
        action last_action() const;

    private:
        std::unique_ptr<handler_impl> impl_;
        action last_action_;
    };

    std::shared_ptr<toolcall::handler> create_handler(const toolcall::params & params);

    class handler_impl {
    public:
        handler_impl(std::string tool_choice)
            : tool_choice_(std::move(tool_choice)) {}

        virtual ~handler_impl() = default;
        virtual std::string tool_list() = 0;
        virtual action call(const std::string & request, std::string & response) = 0;

        const std::string & tool_choice() const { return tool_choice_; }

    protected:
        std::string tool_choice_;
    };

    class loopback_impl : public handler_impl {
    public:
        loopback_impl(std::string tools, std::string tool_choice)
            : handler_impl(tool_choice), tools_(std::move(tools)) {}

        virtual std::string tool_list() override {
            return tools_;
        }

        virtual action call(const std::string & request, std::string & response) override {
            response = request;
            return toolcall::DEFER;
        }

    private:
        std::string tools_;
    };

    class mcp_transport;
    class mcp_impl : public handler_impl {
    public:
        mcp_impl(std::string server_uri, std::string tool_choice);
        mcp_impl(std::vector<std::string> argv, std::string tool_choice);

        virtual std::string tool_list() override;
        virtual action call(const std::string & request, std::string & response) override;

    private:
        std::unique_ptr<mcp_transport> transport_;
    };
}
