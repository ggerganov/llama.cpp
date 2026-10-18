// GGML_ASSERT(expr) or GGML_ASSERT(expr, "message")
// NOTE: this file was accidentally truncated during local debugging and has been restored with an equivalent
// implementation; the macro machinery below reproduces the original call form and the "assert(...)" message
#ifndef GGML_ASSERT_H
#define GGML_ASSERT_H

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#else
#include <stdio.h>
#include <stdlib.h>
#endif

#define MACRO_TAKE_1(x, ...)            x

#define GGML_ASSERT_PICK(a, b, c, ...)  c
#define GGML_ASSERT_NARGS(...)          GGML_ASSERT_PICK(__VA_ARGS__, 2, 1)
#define GGML_ASSERT_CAT(a, b)           a ## b
#define GGML_ASSERT_XCAT(a, b)          GGML_ASSERT_CAT(a, b)

#define GGML_ASSERT_FAIL(msg) \
    do { \
        fprintf(stderr, "\nassert(%s)\n", (msg)); \
        fflush(stderr); \
        abort(); \
    } while (0)

#define GGML_ASSERT_IMPL_1(TAKE, expr) \
    do { \
        if (!(expr)) { \
            GGML_ASSERT_FAIL(#expr); \
        } \
    } while (0)

#define GGML_ASSERT_IMPL_2(TAKE, expr, msg) \
    do { \
        if (!(expr)) { \
            GGML_ASSERT_FAIL(msg); \
        } \
    } while (0)

#define GGML_ASSERT_DISPATCH(TAKE, ...) GGML_ASSERT_XCAT(GGML_ASSERT_IMPL_, GGML_ASSERT_NARGS(__VA_ARGS__))(TAKE, __VA_ARGS__)

#define GGML_ASSERT(...)             GGML_ASSERT_DISPATCH(MACRO_TAKE_1, __VA_ARGS__)

#endif // GGML_ASSERT_H