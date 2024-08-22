from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

n_px = 384
resize_func = Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC, antialias=True)

img_dir = "./imgs"
image_path_1 = f'{img_dir}/image-1d100e9-1.jpg'
image_path_2 = f'{img_dir}/image-1d100e9.jpg'
image_1 = Image.open(image_path_1).convert('RGB')
image_2 = Image.open(image_path_2).convert('RGB')

print(np.asarray(resize_func(image_2))[:5, :10, 0])