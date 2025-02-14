
#include "mcp_stdio_transport.hpp"

toolcall::mcp_stdio_transport::mcp_stdio_transport(std::vector<std::string> argv)
    : argv_(std::move(argv))
{
}

void toolcall::mcp_stdio_transport::start() {
}

void toolcall::mcp_stdio_transport::stop() {
}

bool toolcall::mcp_stdio_transport::send(const mcp::message_variant & /*request*/) {
    return false;
}
