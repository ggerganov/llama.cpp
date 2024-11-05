
import ctypes
import logging
import os

import omni_vlm_cpp


class NexaOmniVlmInference:
    """
    A class used for vision language model inference.
    """

    def __init__(self, llm_model_path: str, mmproj_model_path: str):
        self.llm_model = ctypes.c_char_p(llm_model_path.encode("utf-8"))
        self.mmproj_model = ctypes.c_char_p(mmproj_model_path.encode("utf-8"))

        omni_vlm_cpp.omnivlm_init(self.llm_model, self.mmproj_model)

    def inference(self, prompt: str, image_path: str):
        prompt = ctypes.c_char_p(prompt.encode("utf-8"))
        image_path = ctypes.c_char_p(image_path.encode("utf-8"))
        omni_vlm_cpp.omnivlm_inference(prompt, image_path)

    def __del__(self):
        omni_vlm_cpp.omnivlm_free()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run omni vision language model generation"
    )
    parser.add_argument("--model", type=str, help="Path to the llm model file")
    parser.add_argument("--mmproj", type=str, help="Path to the mmproj file")
    # parser.add_argument("--prompt", type=str, help="prompt string.")
    # parser.add_argument("--image-path", type=str, help="Path to the image.")

    args = parser.parse_args()

    omni_vlm_obj = NexaOmniVlmInference(args.model, args.mmproj)
    # omni_vlm_obj.inference(args.prompt, args.image_path)
    while True:
        print("Input your prompt:")
        prompt = input()
        if prompt == "":
            print("ERROR: you input an empty prompt, try again.")
            continue
        print("Input your image path:")
        image_path = input()
        while not os.path.exists(image_path):
            print("ERROR: can not find image in your input path, please check and input agian.")
            image_path = input()
        omni_vlm_obj.inference(prompt, image_path)
