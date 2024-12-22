#include "llama-arch.h"

#include "llama-impl.h"

LLM_KV::LLM_KV(llm_arch arch) : arch(arch) {}

std::string LLM_KV::operator()(llm_kv kv) const {
    return ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));
}

std::string LLM_TN_IMPL::str() const {
    if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
        return "__missing__";
    }

    std::string name = ::format(LLM_TENSOR_NAMES.at(arch).at(tensor), bid, xid);

    if (suffix != nullptr) {
        name += ".";
        name += suffix;
    }

    return name;
}

const char * llm_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}
