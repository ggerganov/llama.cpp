from enum import IntEnum


class GGML_TYPE(IntEnum):
    """Tensor types, corresponding to enum ggml_type in ggml.h"""

    Q4_0 = 0
    Q4_1 = 1
    I8 = 2
    I16 = 3
    I32 = 4
    F16 = 5
    F32 = 6


class GGML_FILE(IntEnum):
    """File types, corresponding to enum e_ftype in llama.cpp"""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3


ggml_type_from_ftype = {
    GGML_FILE.F32: GGML_TYPE.F32,
    GGML_FILE.F16: GGML_TYPE.F16,
    GGML_FILE.Q4_0: GGML_TYPE.Q4_0,
    GGML_FILE.Q4_1: GGML_TYPE.Q4_1,
}

GGML_BLCK_SIZE = {
    GGML_TYPE.Q4_0: 32,
    GGML_TYPE.Q4_1: 32,
    GGML_TYPE.I8: 1,
    GGML_TYPE.I16: 1,
    GGML_TYPE.I32: 1,
    GGML_TYPE.F16: 1,
    GGML_TYPE.F32: 1,
}

GGML_TYPE_SIZE = {
    GGML_TYPE.Q4_0: 4 + GGML_BLCK_SIZE[GGML_TYPE.Q4_0] // 2,
    GGML_TYPE.Q4_1: 4 * 2 + GGML_BLCK_SIZE[GGML_TYPE.Q4_1] // 2,
    GGML_TYPE.I8: 1,
    GGML_TYPE.I16: 2,
    GGML_TYPE.I32: 4,
    GGML_TYPE.F16: 2,
    GGML_TYPE.F32: 4,
}
