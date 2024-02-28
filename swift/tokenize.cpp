//
// This source file is part of the Stanford Spezi open source project
//
// SPDX-FileCopyrightText: 2022 Stanford University and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

#include "tokenize.h"


/// Tokenize a `String` via a given `llama_context`.
std::vector<llama_token> llama_tokenize_with_context(
     const struct llama_context * ctx,
     const std::string & text,
     bool add_bos,
     bool special) {
    return llama_tokenize(ctx, text, add_bos, special);
}

/// Tokenize a `String` via a given `llama_model`.
std::vector<llama_token> llama_tokenize_with_model(
     const struct llama_model * model,
     const std::string & text,
     bool add_bos,
     bool special) {
    return llama_tokenize(model, text, add_bos, special);
}
