#pragma once

#include "mcp_transport.hpp"

#include <string>
#include <vector>

namespace toolcall
{
    class mcp_stdio_transport : public mcp_transport {
    public:
        mcp_stdio_transport(std::vector<std::string> argv);

        virtual void start() override;
        virtual void stop()  override;
        virtual bool send(const mcp::message_variant & request) override;

    private:
        std::vector<std::string> argv_;
    };
}
