
## Godot Engine Integration: GdNative Library

This is the accompanying code for the Blog post ["Supercharging Game Development with GPU Accelerated Machine Learning"](https://medium.com/@AxSaucedo/supercharging-game-development-with-gpu-accelerated-ml-using-vulkan-kompute-the-godot-game-engine-4e75a84ea9f0). 

This section contains the implementation of the Kompute module as a shared GdNative Library that can be loaded dynamically through the Godot engine. This approach does not require re-compiling the Godot engine source code.


![](https://github.com/KomputeProject/kompute/raw/master/docs/images/komputer-godot-4.gif)

### Set Up Dependencies

We can get all the required dependencies from godot by running

```
git clone --branch 3.2 https://github.com/godotengine/godot-cpp

cd godot-cpp
```

Then we can get all the subomdules

```
git submodule sync
```

and we build the bindings

```
scons -j16 platform=linuxbsd target=debug

```

