//
// This source file is part of the Stanford Spezi open source project
//
// SPDX-FileCopyrightText: 2022 Stanford University and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

#ifndef tokenize_hpp
#define tokenize_hpp

#include <vector>
#include "common.h"


/// Tokenize a `String` via a given `llama_context`.
std::vector<llama_token> llama_tokenize_with_context(
     const struct llama_context * ctx,
     const std::string & text,
     bool add_bos,
     bool special = false);

/// Tokenize a `String` via a given `llama_model`.
std::vector<llama_token> llama_tokenize_with_model(
     const struct llama_model * model,
     const std::string & text,
     bool add_bos,
     bool special = false);


#endif
