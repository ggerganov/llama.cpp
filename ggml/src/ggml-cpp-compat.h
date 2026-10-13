#pragma once

#ifdef __k1om__
#define GGML_IF_CONSTEXPR if
#else
#define GGML_IF_CONSTEXPR if constexpr
#endif

template<typename T>
static inline const T & ggml_clamp(const T & value, const T & low, const T & high) {
    return value < low ? low : high < value ? high : value;
}
