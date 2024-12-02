# Kompute Godot Example

![](https://github.com/KomputeProject/kompute/raw/master/docs/images/komputer-godot-4.gif)

## Set Up Dependencies

We can get all the required dependencies from godot by running

```
git clone --branch 3.2 https://github.com/godotengine/godot-cpp

cd godot-cpp
```

Then we can get all the submodules

```
git submodule sync
```

And we build the bindings

```
scons -j16 platform=linuxbsd target=debug

```
