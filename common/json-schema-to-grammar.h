#pragma once
#include "json.hpp"

std::string tool_call_grammar(const nlohmann::ordered_json & tools);
std::string json_schema_to_grammar(const nlohmann::ordered_json& schema);
