//
// This source file is part of the Stanford Spezi open source project
//
// SPDX-FileCopyrightText: 2022 Stanford University and the project authors (see CONTRIBUTORS.md)
//
// SPDX-License-Identifier: MIT
//

#ifndef vector_hpp
#define vector_hpp

#include <vector>
#include "common.h"


/// Create an empty `vector` of `llama_seq_id`s that serve as a buffer for batch processing.
const std::vector<llama_seq_id> getLlamaSeqIdVector();


#endif
