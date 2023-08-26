#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from embd_input import MyModel
import numpy as np
from torch import nn
import torch

# use PandaGPT path
panda_gpt_path = os.path.join(os.path.dirname(__file__), "PandaGPT")
imagebind_ckpt_path = "./models/panda_gpt/"

sys.path.insert(0, os.path.join(panda_gpt_path,"code","model"))
from ImageBind.models import imagebind_model
from ImageBind import data

ModalityType = imagebind_model.ModalityType
max_tgt_len = 400

class PandaGPT:
    def __init__(self, args):
        self.visual_encoder,_ = imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        self.visual_encoder.eval()
        self.llama_proj = nn.Linear(1024, 5120) # self.visual_hidden_size, 5120)
        self.max_tgt_len = max_tgt_len
        self.model = MyModel(["main", *args])
        self.generated_text = ""
        self.device = "cpu"

    def load_projection(self, path):
        state = torch.load(path, map_location="cpu")
        self.llama_proj.load_state_dict({
            "weight": state["llama_proj.weight"],
            "bias": state["llama_proj.bias"]})

    def eval_inputs(self, inputs):
        self.model.eval_string("<Img>")
        embds = self.extract_multimoal_feature(inputs)
        for i in embds:
            self.model.eval_float(i.T)
        self.model.eval_string("</Img> ")

    def chat(self, question):
        return self.chat_with_image(None, question)

    def chat_with_image(self, inputs, question):
        if self.generated_text == "":
            self.model.eval_string("###")
        self.model.eval_string(" Human: ")
        if inputs:
            self.eval_inputs(inputs)
        self.model.eval_string(question)
        self.model.eval_string("\n### Assistant:")
        ret = self.model.generate_with_print(end="###")
        self.generated_text += ret
        return ret

    def extract_multimoal_feature(self, inputs):
        features = []
        for key in ["image", "audio", "video", "thermal"]:
            if key + "_paths" in inputs:
                embeds = self.encode_data(key, inputs[key+"_paths"])
                features.append(embeds)
        return features

    def encode_data(self, data_type, data_paths):

        type_map = {
            "image": ModalityType.VISION,
            "audio": ModalityType.AUDIO,
            "video": ModalityType.VISION,
            "thermal": ModalityType.THERMAL,
        }
        load_map = {
            "image": data.load_and_transform_vision_data,
            "audio": data.load_and_transform_audio_data,
            "video": data.load_and_transform_video_data,
            "thermal": data.load_and_transform_thermal_data
        }

        load_function = load_map[data_type]
        key = type_map[data_type]

        inputs = {key: load_function(data_paths, self.device)}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            embeds = embeddings[key]
            embeds = self.llama_proj(embeds).cpu().numpy()
        return embeds


if __name__=="__main__":
    a = PandaGPT(["--model", "./models/ggml-vicuna-13b-v0-q4_1.bin", "-c", "2048", "--lora", "./models/panda_gpt/ggml-adapter-model.bin","--temp", "0"])
    a.load_projection("./models/panda_gpt/adapter_model.bin")
    a.chat_with_image(
        {"image_paths": ["./media/llama1-logo.png"]},
        "what is the text in the picture? 'llama' or 'lambda'?")
    a.chat("what is the color of it?")
