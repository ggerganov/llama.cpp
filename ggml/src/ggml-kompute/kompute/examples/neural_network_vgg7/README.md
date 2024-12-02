# Convolutional Neural Network (CNN) VGG7 implementation

This example provides an implementation of a convolutional neural network (CNN) that enables for image resolution upscaling, which means that images can improve their quality through purely the machine learning implementation.

This example demonstrates performing image upscaling using Kompute on the test image below.

In this example we will be doing the following:

* Import pre-trained model
* Create Kompute code that loads model weights
* Create Kompute shader that performs inference on image
* Run model against image to perform upscale

## Import pre-trained model

To import the no-noise-compensation VGG7 model (into `model-kipper`):

```
curl -o model.json https://raw.githubusercontent.com/nagadomi/waifu2x/master/models/vgg_7/art/scale2.0x_model.json
python3 import_vgg7.py model.json
```

Other models from the vgg\_7 set (such as `https://raw.githubusercontent.com/nagadomi/waifu2x/master/models/vgg_7/photo/noise0_model.json`) can be subsituted as desired.

## Create code that loads model weights

We implement the kompute logic under run_vgg7 that loads the model weights and coordinates the execution of the inference.

## Create Kompute shader that performs inference on image

Similarly, we created a compute shader that performs an inference iteration on an image provided to perfrom upscaling.

## Run model against image to perfrom upscale

We now execute model against an image created by us to show how upscaling works. The image used will be the one below:

![](https://raw.githubusercontent.com/KomputeProject/kompute/master/examples/neural_network_vgg7/w2wbinit.png)

To execute that model no tiling is performed, so be careful about image sizes.

We can now run the command below to perform inference against the image blow.

`python3 run_vgg7.py w2wbinit.png out.png`

This would successfully upscale the resolution using the machine learning model, and the result is below:

![](https://raw.githubusercontent.com/KomputeProject/kompute/master/examples/neural_network_vgg7/out.png)


