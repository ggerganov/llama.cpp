
#include <json.hpp>
#include "handler.h"

#ifdef LLAMA_USE_CURL
#    include "mcp_sse_transport.h"
#endif

#include "mcp_stdio_transport.h"

using json = nlohmann::json;

std::shared_ptr<toolcall::handler> toolcall::create_handler(const toolcall::params & params) {
    std::shared_ptr<toolcall::handler> handler;

    auto tools = params.tools();
    auto choice = params.choice();
    if (params.has_uri()) {
#ifdef LLAMA_USE_CURL
        handler.reset(new toolcall::handler(std::make_unique<toolcall::mcp_impl>(tools, choice)));
#endif
    } else {
        handler.reset(new toolcall::handler(std::make_unique<toolcall::loopback_impl>(tools, choice)));
    }

    return handler;
}

std::string toolcall::handler::tool_list() {
    return impl_->tool_list();
}

toolcall::action toolcall::handler::call(const std::string & request, std::string & response) {
    last_action_ = impl_->call(request, response);
    return last_action_;
}

const std::string & toolcall::handler::tool_choice() const {
    return impl_->tool_choice();
}
toolcall::action toolcall::handler::last_action() const {
    return last_action_;
}

#ifdef LLAMA_USE_CURL
toolcall::mcp_impl::mcp_impl(std::string server_uri, std::string tool_choice)
    : handler_impl(tool_choice),
      transport_(new mcp_sse_transport(server_uri))
{
    transport_->start();
}
#else
toolcall::mcp_impl::mcp_impl(std::string /*server_uri*/, std::string tool_choice)
    : handler_impl(tool_choice)
{
}
#endif

toolcall::mcp_impl::mcp_impl(std::vector<std::string> argv, std::string tool_choice)
    : handler_impl(tool_choice),
      transport_(new mcp_stdio_transport(argv))
{
    transport_->start();
}

std::string toolcall::mcp_impl::tool_list() {
    // Construct tools/list call and send to transport
    return json{};// TODO
}

toolcall::action toolcall::mcp_impl::call(const std::string & /*request*/, std::string & /*response*/) {
    // Construct tool call and send to transport
    return toolcall::ACCEPT; // TODO
}
