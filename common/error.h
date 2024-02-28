#pragma once

#include <cstdio>
#include <exception>
#include <string>

class llama_error : public std::exception
{
private:
    std::string _id;
    std::string _description;

public:
    llama_error(const std::string & id, const std::string & description)
    :
    _id(id),
    _description(description)
    {
        fprintf(stderr, "ERROR [%s]: %s\n", id.c_str(), description.c_str());
    }

    inline const std::string & id() const { return _id; }
    inline const std::string & description() const { return _description; }
};
