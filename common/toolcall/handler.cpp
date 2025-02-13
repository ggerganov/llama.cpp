
#include "handler.hpp"

using json = toolcall::json;

toolcall::params::params(std::string tools, std::string choice) {
    this->tools(tools);
    this->choice(choice);
}

static bool starts_with(const std::string & str, const std::string & prefix) {
    return str.size() >= prefix.size()
        && str.compare(0, prefix.size(), prefix) == 0;
}

std::shared_ptr<toolcall::handler> toolcall::create_handler(const toolcall::params & params) {
    std::shared_ptr<toolcall::handler> result;

    auto tools = params.tools();
    auto choice = params.choice();
    bool has_uri = std::holds_alternative<std::string>(tools);
    if (has_uri) {
        auto tools_str = std::get<std::string>(tools);
        result.reset(new toolcall::handler(std::make_unique<toolcall::mcp_impl>(tools_str, choice)));

    } else {
        auto tools_ptr = std::get<toolcall::json_ptr>(tools);
        if (tools_ptr != nullptr) {
            result.reset(new toolcall::handler(std::make_unique<toolcall::loopback_impl>(*tools_ptr, choice)));
        }
    }

    return result;
}

void toolcall::params::tools(std::string tools) {
    try {
        if (tools.empty() || starts_with(tools, "mcp+http")) {
            tools_ = std::move(tools);

        } else {
            tools_ = std::make_shared<json>(json::parse(tools));
            auto tools_ptr = std::get<std::shared_ptr<json>>(tools_);
            if (! tools_ptr->is_array()) {
                throw std::invalid_argument("tools must be a valid JSON array");
            }
        }

    } catch (const json::exception & err) {
        throw std::invalid_argument(err.what());
    }
}

void toolcall::params::choice(std::string choice) {
    try {
        if (choice == "auto" || choice == "required" || choice == "none") {
            tool_choice_ = std::move(choice);

        } else {
            auto choice_ptr = std::make_shared<json>(json::parse(choice));
            tool_choice_ = choice_ptr;
            if (! choice_ptr->is_object()) {
                throw std::invalid_argument(
                    "tool choice must be a valid JSON object, \"auto\", \"required\", or \"none\"");
            }
        }

    } catch (const json::exception & err) {
        throw std::invalid_argument(err.what());
    }
}

toolcall::params::operator bool() const  {
    if (std::holds_alternative<std::string>(tools_)) {
        return ! std::get<std::string>(tools_).empty();

    } else {
        return std::get<toolcall::json_ptr>(tools_) != nullptr;
    }
}

json toolcall::handler::tool_list() {
    return impl_->tool_list();
}

toolcall::action toolcall::handler::call(const json & request, json & response) {
    last_action_ = impl_->call(request, response);
    return last_action_;
}

const toolcall::tool_choice_t & toolcall::handler::tool_choice() const {
    return impl_->tool_choice();
}
toolcall::action toolcall::handler::last_action() const {
    return last_action_;
}

toolcall::mcp_impl::mcp_impl(std::string server_uri, tool_choice_t tool_choice)
    : handler_impl(tool_choice)
{
    // TODO
}

json toolcall::mcp_impl::tool_list() {
    return json{};// TODO
}

toolcall::action toolcall::mcp_impl::call(const json & request, json & response) {
    return toolcall::ACCEPT; // TODO
}
