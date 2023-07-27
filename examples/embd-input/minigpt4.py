import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from embd_input import MyModel
import numpy as np
from torch import nn
import torch
from PIL import Image

minigpt4_path = os.path.join(os.path.dirname(__file__), "MiniGPT-4")
sys.path.insert(0, minigpt4_path)
from minigpt4.models.blip2 import Blip2Base
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor


class MiniGPT4(Blip2Base):
    """
    MiniGPT4 model from https://github.com/Vision-CAIR/MiniGPT-4
    """
    def __init__(self,
        args,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0
    ):
        super().__init__()
        self.img_size = img_size
        self.low_resource = low_resource
        self.preprocessor = Blip2ImageEvalProcessor(img_size)

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        print('Loading VIT Done')
        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        print('Loading Q-Former Done')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, 5120 # self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.model = MyModel(["main", *args])
        # system prompt
        self.model.eval_string("Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions."
           "###")

    def encode_img(self, image):
        image = self.preprocessor(image)
        image = image.unsqueeze(0)
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            # atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama

    def load_projection(self, path):
        state = torch.load(path)["model"]
        self.llama_proj.load_state_dict({
            "weight": state["llama_proj.weight"],
            "bias": state["llama_proj.bias"]})

    def chat(self, question):
        self.model.eval_string("Human: ")
        self.model.eval_string(question)
        self.model.eval_string("\n### Assistant:")
        return self.model.generate_with_print(end="###")

    def chat_with_image(self, image, question):
        with torch.no_grad():
            embd_image = self.encode_img(image)
        embd_image = embd_image.cpu().numpy()[0]
        self.model.eval_string("Human: <Img>")
        self.model.eval_float(embd_image.T)
        self.model.eval_string("</Img> ")
        self.model.eval_string(question)
        self.model.eval_string("\n### Assistant:")
        return self.model.generate_with_print(end="###")


if __name__=="__main__":
    a = MiniGPT4(["--model", "./models/ggml-vicuna-13b-v0-q4_1.bin", "-c", "2048"])
    a.load_projection(os.path.join(
        os.path.dirname(__file__) ,
        "pretrained_minigpt4.pth"))
    respose = a.chat_with_image(
        Image.open("./media/llama1-logo.png").convert('RGB'),
        "what is the text in the picture?")
    a.chat("what is the color of it?")
