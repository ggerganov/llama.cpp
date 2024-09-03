# from torchvision.transforms import Resize
# from torchvision.transforms import InterpolationMode
# from PIL import Image
# import numpy as np

# n_px = 384
# resize_func = Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True)

# img_dir = "./imgs"
# image_path_1 = f'{img_dir}/image-1d100e9-1.jpg'
# image_path_2 = f'{img_dir}/image-1d100e9.jpg'
# image_1 = Image.open(image_path_1).convert('RGB')
# image_2 = Image.open(image_path_2).convert('RGB')

# print(np.asarray(resize_func(image_2))[:5, :10, 0])


import gguf
import numpy as np
import torch

patches_embeddings = torch.load('./imgs/4patches_embeddings.pt').numpy()
print(f'4patches_embeddings:{patches_embeddings.shape}\n')
print(patches_embeddings[1:,:,:])


# gguf_writer = gguf.GGUFWriter(path='./imgs/4patches_embeddings.gguf', arch='4patches_embeddings')
# gguf_writer.add_tensor("data", patches_embeddings)
# gguf_writer.write_header_to_file()
# gguf_writer.write_kv_data_to_file()
# gguf_writer.write_tensors_to_file()
# gguf_writer.close()