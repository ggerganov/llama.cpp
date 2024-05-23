from __future__ import annotations


def model_weight_count_rounded_notation(model_params_count: int) -> str:
    if model_params_count > 1e15 :
        # Quadrillion Of Parameters
        scaled_model_params = model_params_count * 1e-15
        scale_suffix = "Q"
    elif model_params_count > 1e12 :
        # Trillions Of Parameters
        scaled_model_params = model_params_count * 1e-12
        scale_suffix = "T"
    elif model_params_count > 1e9 :
        # Billions Of Parameters
        scaled_model_params = model_params_count * 1e-9
        scale_suffix = "B"
    elif model_params_count > 1e6 :
        # Millions Of Parameters
        scaled_model_params = model_params_count * 1e-6
        scale_suffix = "M"
    else:
        # Thousands Of Parameters
        scaled_model_params = model_params_count * 1e-3
        scale_suffix = "K"
    return f"{round(scaled_model_params)}{scale_suffix}"


def naming_convention(model_name: str, version_string:str, expert_count_int:int, model_params_count: int, encodingScheme: str) -> str:
    # Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#gguf-naming-convention
    name = model_name.strip().replace(' ', '-') if model_name is not None else "ggml-model"
    version = f"-{version_string}" if version_string is not None else ""
    expert_count_chunk = f"{expert_count_int}x" if expert_count_int is not None and expert_count_int > 0 else ""
    parameters = model_weight_count_rounded_notation(model_params_count)
    encodingScheme = encodingScheme.upper()
    return f"{name}{version}-{expert_count_chunk}{parameters}-{encodingScheme}"
