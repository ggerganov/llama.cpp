#pragma once

#include <string>
#include <variant>
#include <memory>

#include "../json.hpp"

namespace toolcall
{
    using json = nlohmann::ordered_json;
    using json_ptr = std::shared_ptr<json>;
    using tools_t = std::variant<std::string, json_ptr>;
    using tool_choice_t = std::variant<std::string, json_ptr>;

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

        json tool_list();
        action call(const json & request, json & response);
        const tool_choice_t & tool_choice() const;
        action last_action() const;

    private:
        std::unique_ptr<handler_impl> impl_;
        action last_action_;
    };

    class params {
    public:
        params(std::string tools = "", std::string choice = "auto");

        params(const params & other) = default;
        params(params && other) noexcept = default;
        params & operator=(const params & other) = default;
        params & operator=(params && other) noexcept = default;

        operator bool() const;

        void tools(std::string tools);
        const tools_t tools() const { return tools_; }

        void choice(std::string choice);
        const tool_choice_t & choice() const { return tool_choice_; }

    private:
        tools_t tools_;
        tool_choice_t tool_choice_;
    };

    std::shared_ptr<toolcall::handler> create_handler(const toolcall::params & params);

    class handler_impl {
    public:
        handler_impl(tool_choice_t tool_choice)
            : tool_choice_(std::move(tool_choice)) {}

        virtual ~handler_impl() = default;
        virtual json tool_list() = 0;
        virtual action call(const json & request, json & response) = 0;

        const tool_choice_t & tool_choice() const { return tool_choice_; }

    protected:
        tool_choice_t tool_choice_;
    };

    class loopback_impl : public handler_impl {
    public:
        loopback_impl(json tools, tool_choice_t tool_choice)
            : handler_impl(tool_choice), tools_(std::move(tools)) {}

        virtual json tool_list() override {
            return tools_;
        }

        virtual action call(const json & request, json & response) override {
            response = request;
            return toolcall::DEFER;
        }

    private:
        json tools_;
    };

    class mcp_transport;
    class mcp_impl : public handler_impl {
    public:
        mcp_impl(std::string server_uri, tool_choice_t tool_choice);
        mcp_impl(std::vector<std::string> argv, tool_choice_t tool_choice);

        virtual json tool_list() override;
        virtual action call(const json & request, json & response) override;

    private:
        std::unique_ptr<mcp_transport> transport_;
    };
}
