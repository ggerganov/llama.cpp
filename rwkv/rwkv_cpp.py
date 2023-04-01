import os
import ctypes
import torch
import multiprocessing
from typing import Tuple, Optional

P_FLOAT = ctypes.POINTER(ctypes.c_float)

class RWKVModel:
    """
    PyTorch wrapper around rwkv.cpp shared library.
    """

    def __init__(
            self,
            shared_library_path: str,
            model_path: str,
            thread_count: int = max(1, multiprocessing.cpu_count() // 2)
    ):
        """
        Loads the model and prepares it for inference.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library_path : str
            Path to rwkv.cpp shared library. On Windows, it would look like 'rwkv.dll'. On UNIX, 'rwkv.so'.
        model_path : str
            Path to RWKV model file in ggml format.
        thread_count : int
            Thread count to use. If not set, defaults to CPU count / 2.
        """

        assert os.path.isfile(shared_library_path), f'{shared_library_path} is not a file'
        assert os.path.isfile(model_path), f'{model_path} is not a file'
        assert thread_count > 0, 'Thread count must be positive'

        self.library = ctypes.cdll.LoadLibrary(shared_library_path)

        self.library.rwkv_init_from_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.library.rwkv_init_from_file.restype = ctypes.c_void_p

        self.library.rwkv_eval.argtypes = [
            ctypes.c_void_p, # ctx
            ctypes.c_long, # token
            P_FLOAT, # state_in
            P_FLOAT, # state_out
            P_FLOAT  # logits_out
        ]
        self.library.rwkv_eval.restype = ctypes.c_bool

        self.library.rwkv_get_state_buffer_element_count.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_state_buffer_element_count.restype = ctypes.c_size_t

        self.library.rwkv_get_logits_buffer_element_count.argtypes = [ctypes.c_void_p]
        self.library.rwkv_get_logits_buffer_element_count.restype = ctypes.c_size_t

        self.library.rwkv_free.argtypes = [ctypes.c_void_p]
        self.library.rwkv_free.restype = None

        self.ctx = self.library.rwkv_init_from_file(model_path.encode('utf-8'), ctypes.c_int(thread_count))

        assert self.ctx is not None, 'Failed to load the model, see stderr'

        self.state_buffer_element_count = self.library.rwkv_get_state_buffer_element_count(self.ctx)
        self.logits_buffer_element_count = self.library.rwkv_get_logits_buffer_element_count(self.ctx)

        self.valid = True

    def eval(self, token: int, state_in: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the model for a single token.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        token : int
            Index of next token to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[torch.Tensor]
            State from previous call of this method. If this is a first pass, set it to None.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        assert self.valid, 'Model was freed'

        if state_in is None:
            state_in_ptr = 0
        else:
            expected_shape = (self.state_buffer_element_count,)

            assert state_in.is_contiguous(), 'State tensor is not contiguous'
            assert state_in.shape == expected_shape, f'Invalid state shape {state_in.shape}, expected {expected_shape}'

            state_in_ptr = state_in.storage().data_ptr()

        # TODO Probably these allocations can be optimized away
        state_out: torch.Tensor = torch.zeros(self.state_buffer_element_count, dtype=torch.float32, device='cpu')
        logits_out: torch.Tensor = torch.zeros(self.logits_buffer_element_count, dtype=torch.float32, device='cpu')

        result = self.library.rwkv_eval(
            self.ctx,
            ctypes.c_long(token),
            ctypes.cast(state_in_ptr, P_FLOAT),
            ctypes.cast(state_out.storage().data_ptr(), P_FLOAT),
            ctypes.cast(logits_out.storage().data_ptr(), P_FLOAT)
        )

        assert result, 'Inference failed, see stderr'

        return logits_out, state_out

    def free(self):
        """
        Frees all allocated resources.
        In case of any error, this method will throw an exception.
        The object must not be used anymore after calling this method.
        """

        assert self.valid, 'Already freed'

        self.valid = False

        self.library.rwkv_free(self.ctx)
