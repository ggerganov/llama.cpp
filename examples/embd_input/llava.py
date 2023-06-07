import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from embd_input import MyModel
import numpy as np
from torch import nn
import torch
from transformers import CLIPVisionModel,  CLIPImageProcessor
from PIL import Image
vision_tower = "openai/clip-vit-large-patch14"

class Llava:
    def __init__(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        self.mm_projector = nn.Linear(1024, 5120)
        self.model = MyModel(["main", "--model", "../llama.cpp/models/ggml-vic13b-q4_1.bin", "-c", "2048"])

    def chat_with_image(self, image, question):
        with torch.no_grad():
            embd_image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_forward_out = self.vision_tower(embd_image.unsqueeze(0), output_hidden_states=True)
            select_hidden_state_layer = -2
            select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
            image_feature = select_hidden_state[:, 1:]
            embd_image = self.mm_projector(image_feature)
            embd_image = embd_image.cpu().numpy()
        self.model.eval_string("user: ")
        # print(embd_image.shape)
        self.model.eval_float(embd_image.T)
        self.model.eval_string(question)
        self.model.eval_string("\nassistant: ")
        ret = ""
        for _ in range(500):
            tmp = self.model.sampling().decode()
            if tmp == "":
                break
            ret += tmp
        return ret

a = Llava()
state = torch.load(os.path.dirname(__file__) + "/a.pth")
a.mm_projector.load_state_dict({"weight": state["model.mm_projector.weight"], "bias": state["model.mm_projector.bias"]})
print(a.chat_with_image(Image.open("./media/llama1-logo.png").convert('RGB'), "what is the text in the picture?"))


