#pragma once

#include <cstdio>
#include <exception>
#include <string>

class llama_error : public std::exception
{
private:
    std::string _type;
    std::string _message;

public:
    llama_error(const std::string & type, const std::string & message)
    :
    _type(type),
    _message(message)
    {
        fprintf(stderr, "ERROR [%s]: %s\n", type.c_str(), message.c_str());
    }

    inline const std::string & type() const { return _type; }
    inline const std::string & message() const { return _message; }
};
