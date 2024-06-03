from __future__ import annotations


def fill_templated_filename(filename: str, output_type: str):
    # Given a file name fill in any type templates e.g. 'some-model-name.{ftype}.gguf'
    ftype_uppercase: str = output_type.upper()
    ftype_lowercase: str = output_type.lower()
    return filename.format(ftype_lowercase,
                           outtype=ftype_lowercase, ftype=ftype_lowercase,
                           OUTTYPE=ftype_uppercase, FTYPE=ftype_uppercase)


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


def parameter_weight_class(expert_count_int:int, model_params_count: int) -> str:
    per_model_rounded_weight_estimate = model_weight_count_rounded_notation(model_params_count)

    if expert_count_int is not None and expert_count_int > 0:
        size_class = f"{expert_count_int}x{per_model_rounded_weight_estimate}"
    else:
        size_class = f"{per_model_rounded_weight_estimate}"

    return size_class

def naming_convention(model_name: str, base_name: str, finetune_string:str, version_string:str, parameter_weight_class: str, output_type: str) -> str:
    # Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#gguf-naming-convention

    if base_name is not None:
        name = base_name.strip().title().replace(' ', '-').replace('/', '-')
    elif model_name is not None:
        name = model_name.strip().title().replace(' ', '-').replace('/', '-')
    else:
        name = "ggml-model"

    parameters = f"-{parameter_weight_class}" if parameter_weight_class is not None else ""

    finetune = f"-{finetune_string.strip().title().replace(' ', '-')}" if finetune_string is not None else ""

    version = f"-{version_string.strip().replace(' ', '-')}" if version_string is not None else ""

    precision = f"-{output_type.strip().replace(' ', '-').upper()}"

    return f"{name}{parameters}{finetune}{version}{precision}"
