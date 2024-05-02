#pragma once
#include "json.hpp"

std::string tool_call_grammar(const nlohmann::ordered_json & tools, bool allow_parallel_calls = false);
std::string json_schema_to_grammar(const nlohmann::ordered_json& schema);
