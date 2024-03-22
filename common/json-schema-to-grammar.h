#pragma once
#include "json.hpp"

std::string json_schema_to_grammar(const nlohmann::ordered_json& schema);
