## KoboldCpp based GGML Backend by Concedo
## For use as a custom backend in KoboldAI United
## Not intended for general use.

from __future__ import annotations

import time, json
import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os
from . import koboldcpp

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)

model_backend_name = "koboldcpp" #specific instead of ggml
model_backend_type = "ggml" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

kcpp_backend_loaded = False

class KoboldCppException(Exception):
    """To be used for errors on cpp side of KoboldCpp."""

class KcppArgsObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class model_backend(InferenceModel):
    def __init__(self) -> None:
        super().__init__()

    def is_valid(self, model_name, model_path, menu_path):
        return "ggml" in model_name.lower()

    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):
        self.filename = model_name #model_path is null, name is path for some reason
        self.model_name = "GGML_Model"
        try:
            from pathlib import Path
            self.model_name = Path(model_name).name
        except:
            pass
        requested_parameters = []
        return requested_parameters

    def set_input_parameters(self, parameters):
        pass

    def _load(self, save_model: bool, initial_load: bool) -> None:
        global kcpp_backend_loaded
        self.tokenizer = self._get_tokenizer("gpt2")
        if not kcpp_backend_loaded:
            kcppargs = KcppArgsObject(model=self.filename, model_param=self.filename,
            port=5001, port_param=5001, host='', launch=False, lora=None, threads=5, blasthreads=5,
            psutil_set_threads=False, highpriority=False, contextsize=2048,
            blasbatchsize=512, ropeconfig=[0.0, 10000.0], stream=False, smartcontext=False,
            unbantokens=False, bantokens=None, usemirostat=None, forceversion=0, nommap=False,
            usemlock=False, noavx2=False, debugmode=0, skiplauncher=False, hordeconfig=None, noblas=False,
            useclblast=None, usecublas=None, gpulayers=0, tensor_split=None, config=None)

            koboldcpp.main(kcppargs,False) #initialize library without enabling Lite http server
            kcpp_backend_loaded = True
        pass

    def _save_settings(self):
        pass

    def _raw_generate(
        self,
        prompt_tokens: Union[List[int], torch.Tensor],
        max_new: int,
        gen_settings: GenerationSettings,
        single_line: bool = False,
        batch_count: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:

        decoded_prompt = utils.decodenewlines(self.tokenizer.decode(prompt_tokens))

        # Store context in memory to use it for comparison with generated content
        utils.koboldai_vars.lastctx = decoded_prompt

        genresult = koboldcpp.generate(decoded_prompt,max_new,utils.koboldai_vars.max_length,
        gen_settings.temp,int(gen_settings.top_k),gen_settings.top_a,gen_settings.top_p,
        gen_settings.typical,gen_settings.tfs,gen_settings.rep_pen,gen_settings.rep_pen_range)

        outputs = [genresult]
        return GenerationResult(
            model=self,
            out_batches=np.array(
                [self.tokenizer.encode(x) for x in outputs]
            ),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )
