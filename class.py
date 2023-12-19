## KoboldCpp based GGML Backend by Concedo
## For use as a custom backend in KoboldAI United
## Not intended for general use.

from __future__ import annotations

import time, json
import torch
import requests
import numpy as np
from typing import List, Optional, Union
import os, time
from . import koboldcpp

import utils
from logger import logger
from modeling.inference_model import (
    GenerationResult,
    GenerationSettings,
    InferenceModel,
)

model_backend_name = "KoboldCPP" #specific instead of ggml
model_backend_type = "ggml" #This should be a generic name in case multiple model backends are compatible (think Hugging Face Custom and Basic Hugging Face)

class KoboldCppException(Exception):
    """To be used for errors on cpp side of KoboldCpp."""

class KcppArgsObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class model_backend(InferenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.kcpp_backend_loaded = False

    def is_valid(self, model_name, model_path, menu_path):

        foundfile = False
        try:
            files = os.listdir(model_path)
            foundfile = len([filename for filename in files if (("ggml" in filename.lower() and ".bin" in filename.lower()) or ".gguf" in filename.lower())])>0
        except:
            pass
        return foundfile

    def get_requested_parameters(self, model_name, model_path, menu_path, parameters = {}):

        self.kcpp_threads = 5
        self.model_name = "GGML_Model"
        self.kcpp_ctxsize = 2048
        self.kcpp_blasbatchsize = 512
        self.kcpp_gpulayers = 0
        self.kcpp_smartcontext = False
        self.kcpp_ropescale = 0.0
        self.kcpp_ropebase = 10000.0
        self.kcpp_useclblast = None
        self.kcpp_usecublas = None
        self.kcpp_noblas = False
        self.kcpp_noavx2 = False
        self.kcpp_nommap = False
        self.kcpp_debugmode = 0
        self.kcpp_tensor_split_str = ""
        self.kcpp_tensor_split = None

        files = os.listdir(model_path)
        foundfiles = [filename for filename in files if (("ggml" in filename.lower() and ".bin" in filename.lower()) or ".gguf" in filename.lower())]

        requested_parameters = []
        foldermdls = []
        for ff in foundfiles:
            foldermdls.append({'text': ff, 'value': os.path.join(model_path, ff)})
        requested_parameters.append({
                                    "uitype": "dropdown",
                                    "unit": "string",
                                    "label": "GGML DataFile Name",
                                    "id": "kcpp_filename",
                                    "default": os.path.join(model_path, foundfiles[0]) if len(foundfiles)>0 else model_name,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "Actual GGML DataFile Name",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": "",
                                    'children': foldermdls
                                    })
        requested_parameters.append({
                                    "uitype": "dropdown",
                                    "unit": "int",
                                    "label": "KoboldCpp Accelerator",
                                    "id": "kcpp_accelerator",
                                    "default": 0,
                                    "check": {"value": "", 'check': "!="},
                                    'multiple': False,
                                    "tooltip": "KoboldCpp Accelerator",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": "",
                                    'children': [{'text': 'Use No BLAS', 'value': 0}, {'text': 'Use OpenBLAS', 'value': 1}, {'text': 'Use CuBLAS', 'value': 2},
                                    {'text': 'Use CLBLast GPU #1', 'value': 3},{'text': 'Use CLBLast GPU #2', 'value': 4},{'text': 'Use CLBLast GPU #3', 'value': 5}
                                    ,{'text': 'NoAVX2 Mode (Old CPU)', 'value': 6},{'text': 'Failsafe Mode (Old CPU)', 'value': 7}],
                                    })
        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "int",
                                    "label": "Threads",
                                    "id": "kcpp_threads",
                                    "default": self.kcpp_threads,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "Thread Count",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })

        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "int",
                                    "label": "Max Context Size",
                                    "id": "kcpp_ctxsize",
                                    "default": self.kcpp_ctxsize,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "Max Context Size",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })
        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "int",
                                    "label": "BLAS Batch Size",
                                    "id": "kcpp_blasbatchsize",
                                    "default": self.kcpp_blasbatchsize,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "BLAS Batch Size",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })
        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "int",
                                    "label": "GPU Layers",
                                    "id": "kcpp_gpulayers",
                                    "default": self.kcpp_gpulayers,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "GPU Layers",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })
        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "int",
                                    "label": "Rope Scale",
                                    "id": "kcpp_ropescale",
                                    "default": self.kcpp_ropescale,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "Rope Scale",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })
        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "int",
                                    "label": "Rope Base",
                                    "id": "kcpp_ropebase",
                                    "default": self.kcpp_ropebase,
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "Rope Base",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })
        requested_parameters.append({
                                    "uitype": "dropdown",
                                    "unit": "int",
                                    "label": "Smart Context",
                                    "id": "kcpp_smartcontext",
                                    "default": self.kcpp_smartcontext,
                                    "check": {"value": "", 'check': "!="},
                                    'multiple': False,
                                    "tooltip": "Smart Context",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": "",
                                    'children': [{'text': 'False', 'value': False}, {'text': 'True', 'value': True}],
                                    })
        requested_parameters.append({
                                    "uitype": "text",
                                    "unit": "text",
                                    "label": "GPU ID",
                                    "id": "kcpp_tensor_split_str",
                                    "default": "1",
                                    "check": {"value": "", 'check': "!="},
                                    "tooltip": "Which GPU's do we use? For example:1 2",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": ""
                                    })
        requested_parameters.append({
                                    "uitype": "dropdown",
                                    "unit": "int",
                                    "label": "Debug Mode",
                                    "id": "kcpp_debugmode",
                                    "default": self.kcpp_debugmode,
                                    "check": {"value": "", 'check': "!="},
                                    'multiple': False,
                                    "tooltip": "Debug Mode",
                                    "menu_path": "",
                                    "refresh_model_inputs": False,
                                    "extra_classes": "",
                                    'children': [{'text': 'False', 'value': 0}, {'text': 'True', 'value': 1}],
                                    })
        return requested_parameters

    def set_input_parameters(self, parameters):
        self.kcpp_threads = parameters["kcpp_threads"]
        self.kcpp_filename = parameters["kcpp_filename"]
        self.kcpp_ctxsize = parameters["kcpp_ctxsize"]
        self.kcpp_blasbatchsize = parameters["kcpp_blasbatchsize"]
        self.kcpp_gpulayers = parameters["kcpp_gpulayers"]
        self.kcpp_smartcontext = parameters["kcpp_smartcontext"]
        self.kcpp_ropescale = parameters["kcpp_ropescale"]
        self.kcpp_ropebase = parameters["kcpp_ropebase"]
        self.kcpp_debugmode = parameters["kcpp_debugmode"]
        self.kcpp_tensor_split_str = parameters["kcpp_tensor_split_str"]
        if self.kcpp_tensor_split_str and self.kcpp_tensor_split_str!="":
            splits = self.kcpp_tensor_split_str.split()
            self.kcpp_tensor_split = []
            for s in splits:
                self.kcpp_tensor_split.append(int(s))
                print(self.kcpp_tensor_split)

        accel = parameters["kcpp_accelerator"]
        if accel==0:
            self.kcpp_noblas = True
        elif accel==1:
           pass
        elif accel==2:
            self.kcpp_usecublas = ["normal"]
        elif accel==3:
            self.kcpp_useclblast = [0,0]
        elif accel==4:
            self.kcpp_useclblast = [1,0]
        elif accel==5:
            self.kcpp_useclblast = [0,1]
        elif accel==6:
            self.kcpp_noavx2 = True
        elif accel==7:
            self.kcpp_noavx2 = True
            self.kcpp_noblas = True
            self.kcpp_nommap = True
        pass

    def unload(self):
        print("Attemping to unload library")
        self.process.terminate()
        

    def _load(self, save_model: bool, initial_load: bool) -> None:
        self.tokenizer = self._get_tokenizer("gpt2")
        kcppargs = KcppArgsObject(model=self.kcpp_filename, model_param=self.kcpp_filename,
        port=5001, port_param=5001, host='', launch=False, lora=None, threads=self.kcpp_threads, blasthreads=self.kcpp_threads,
        psutil_set_threads=False, highpriority=False, contextsize=self.kcpp_ctxsize,
        blasbatchsize=self.kcpp_blasbatchsize, ropeconfig=[self.kcpp_ropescale, self.kcpp_ropebase], stream=False, smartcontext=self.kcpp_smartcontext,
        unbantokens=False, bantokens=None, usemirostat=None, forceversion=0, nommap=self.kcpp_nommap,
        usemlock=False, noavx2=self.kcpp_noavx2, debugmode=self.kcpp_debugmode, skiplauncher=True, hordeconfig=None, noblas=self.kcpp_noblas,
        useclblast=self.kcpp_useclblast, usecublas=self.kcpp_usecublas, gpulayers=self.kcpp_gpulayers, tensor_split=self.kcpp_tensor_split, config=None,
        onready='', multiuser=False, foreground=False, preloadstory=None, noshift=False, remotetunnel=False)
        

        #koboldcpp.main(kcppargs,False) #initialize library without enabling Lite http server
        (self.output_queue, self.input_queue, self.process) = koboldcpp.start_in_seperate_process(kcppargs)
        while True:
            data = self.output_queue.get()
            if data['command'] == 'load status':
                utils.koboldai_vars.total_layers = data['data']['total']
                utils.koboldai_vars.loaded_layers = data['data']['loaded']
            elif data['command'] == 'complete':
                break
            time.sleep(0.02)

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

        self.input_queue.put({'command': 'generate', 'data': [(decoded_prompt,max_new,utils.koboldai_vars.max_length,
                                gen_settings.temp,int(gen_settings.top_k),gen_settings.top_a,gen_settings.top_p,
                                gen_settings.typical,gen_settings.tfs,gen_settings.rep_pen,gen_settings.rep_pen_range),
                               {"sampler_order": gen_settings.sampler_order, "use_default_badwordsids": utils.koboldai_vars.use_default_badwordsids}
                                ]})
        
        #genresult = koboldcpp.generate(decoded_prompt,max_new,utils.koboldai_vars.max_length,
        #gen_settings.temp,int(gen_settings.top_k),gen_settings.top_a,gen_settings.top_p,
        #gen_settings.typical,gen_settings.tfs,gen_settings.rep_pen,gen_settings.rep_pen_range,
        #sampler_order=gen_settings.sampler_order,use_default_badwordsids=utils.koboldai_vars.use_default_badwordsids)
           
        genresult = []
        while True:
            data = self.output_queue.get()
            print(data)
            if data['command'] == 'generated text':
                genresult.append(data['data'])
                if self.output_queue.empty():
                    break
            time.sleep(0.02)

        return GenerationResult(
            model=self,
            out_batches=np.array(
                [self.tokenizer.encode(x) for x in genresult]
            ),
            prompt=prompt_tokens,
            is_whole_generation=True,
            single_line=single_line,
        )
