import ctypes
import os
import sys
from pathlib import Path


# Load the library
def _load_shared_library(lib_base_name: str, base_path: Path = None):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".dylib"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    if base_path is None:
        _base_path = Path(__file__).parent.parent.resolve()
    else:
        print(f"Using base path: {base_path}")
        _base_path = base_path
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        print(f"Trying to load shared library '{_lib_path}'")
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                print(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "omni_vlm_wrapper_shared"
base_path = (
    Path(__file__).parent.parent.parent.resolve()
    / "build"
    / "examples"
    / "omni-vlm"
)

# Load the library
_lib = _load_shared_library(_lib_base_name, base_path)

omni_char_p = ctypes.c_char_p


def omnivlm_init(llm_model_path: omni_char_p, mmproj_model_path: omni_char_p, vlm_version: omni_char_p):
    return _lib.omnivlm_init(llm_model_path, mmproj_model_path, vlm_version)


_lib.omnivlm_init.argtypes = [omni_char_p, omni_char_p, omni_char_p]
_lib.omnivlm_init.restype = None


def omnivlm_inference(prompt: omni_char_p, image_path: omni_char_p):
    return _lib.omnivlm_inference(prompt, image_path)


_lib.omnivlm_inference.argtypes = [omni_char_p, omni_char_p]
_lib.omnivlm_inference.restype = omni_char_p


def omnivlm_free():
    return _lib.omnivlm_free()


_lib.omnivlm_free.argtypes = []
_lib.omnivlm_free.restype = None
