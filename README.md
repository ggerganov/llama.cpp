# llamacpypy
llamacpp but wrapped in python

This allows serving llama using libraries such as fastAPI using the optimized and in particular quantized models of the [llama.cpp](https://github.com/ggerganov/llama.cpp) ecosystem instead of using torch directly. This should decrease ressource consumption over plain torch.

# Installation

Atm this is all very raw so it will require some work on the users part.

### Clone the repo and pull the external repo for pybind11

```
git clone https://github.com/seemanne/llamacpypy.git
cd llamacpypy
git submodule update --init
```

### Set up your venv and install the requirements as always

If you have poetry, there are artifacts in the pyproject file that should allow you to do `poetry install` to set up venv, however it wont install the project itself. This can be done by using `poetry shell` and then calling `pip install ./` as below.

If anyone want to fix the build process to make it less cumbersome, I would be very happy.

If you have another setup just pip install the reqs in your virtual env of choice and then continue as described below. 

### Run makefile 

This isn't actually required, but it will give compile errors if something is wrong.
```
make -j
```

### Install the module using pip 

```
pip install ./
```

# Usage

Initialize the model instance:
```
from llamacpypy import Llama

llama = Llama('models/7B/ggml-model-q4_0.bin', warm_start=False)
```
Load your model into memory:
```
llama.load_model()
```
Generate from a given prompt:
```
var = llama.generate("This is the weather report, we are reporting a clown fiesta happening at backer street. The clowns ")
print(var)
>>> This is the weather report, we are reporting a clown fiesta happening at backer street. The clowns 1st of July parade was going to be in their own neighborhood but they just couldn't contain themselves;
They decided it would look better and probably have more fun if all went into one area which meant that the whole town had to shut down for a little while as all roads were blocked. At least traffic wasn’t too bad today because most of people are out shopping, but I did see some shoppers in their car driving away from Backer street with “clowns” on wheels outside their windows…
The kids lined up along the route and waited for the parade to pass by
```

# Implementation details

This python module is mainly a wrapper around the `llama` class in `src/inference.cpp`. As such, any changes should be done in there. 
As the llamacpp code is mostly contained in `main.cpp` which doesn't expose a good api, this repo will have to be manually patched on a need-be basis. Changes to `ggml` should not be a problem. Fixing the api on the main repo would allow this to be set up as a downstream fork rather than the weird sidekick repo it currently is.