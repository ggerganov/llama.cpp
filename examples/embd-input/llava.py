#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from embd_input import MyModel
import numpy as np
from torch import nn
import torch
from transformers import CLIPVisionModel,  CLIPImageProcessor
from PIL import Image

# model parameters from 'liuhaotian/LLaVA-13b-delta-v1-1'
vision_tower = "openai/clip-vit-large-patch14"
select_hidden_state_layer = -2
# (vision_config.image_size // vision_config.patch_size) ** 2
image_token_len = (224//14)**2

class Llava:
    def __init__(self, args):
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        self.mm_projector = nn.Linear(1024, 5120)
        self.model = MyModel(["main", *args])

    def load_projection(self, path):
        state = torch.load(path)
        self.mm_projector.load_state_dict({
            "weight": state["model.mm_projector.weight"],
            "bias": state["model.mm_projector.bias"]})

    def chat(self, question):
        self.model.eval_string("user: ")
        self.model.eval_string(question)
        self.model.eval_string("\nassistant: ")
        return self.model.generate_with_print()

    def chat_with_image(self, image, question):
        with torch.no_grad():
            embd_image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_forward_out = self.vision_tower(embd_image.unsqueeze(0), output_hidden_states=True)
            select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
            image_feature = select_hidden_state[:, 1:]
            embd_image = self.mm_projector(image_feature)
            embd_image = embd_image.cpu().numpy()[0]
        self.model.eval_string("user: ")
        self.model.eval_token(32003-2) # im_start
        self.model.eval_float(embd_image.T)
        for i in range(image_token_len-embd_image.shape[0]):
            self.model.eval_token(32003-3) # im_patch
        self.model.eval_token(32003-1) # im_end
        self.model.eval_string(question)
        self.model.eval_string("\nassistant: ")
        return self.model.generate_with_print()


if __name__=="__main__":
    # model form liuhaotian/LLaVA-13b-delta-v1-1
    a = Llava(["--model", "./models/ggml-llava-13b-v1.1.bin", "-c", "2048"])
    # Extract from https://huggingface.co/liuhaotian/LLaVA-13b-delta-v1-1/blob/main/pytorch_model-00003-of-00003.bin.
    # Also here can use pytorch_model-00003-of-00003.bin directly.
    a.load_projection(os.path.join(
        os.path.dirname(__file__) ,
        "llava_projection.pth"))
    respose = a.chat_with_image(
        Image.open("./media/llama1-logo.png").convert('RGB'),
        "what is the text in the picture?")
    respose
    a.chat("what is the color of it?")



