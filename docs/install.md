# Install pre-built version of jarvis.cpp

## Homebrew

On Mac and Linux, the homebrew package manager can be used via

```sh
brew install jarvis.cpp
```
The formula is automatically updated with new `jarvis.cpp` releases. More info: https://github.com/ggerganov/jarvis.cpp/discussions/7668

## Nix

On Mac and Linux, the Nix package manager can be used via

```sh
nix profile install nixpkgs#jarvis-cpp
```
For flake enabled installs.

Or

```sh
nix-env --file '<nixpkgs>' --install --attr jarvis-cpp
```

For non-flake enabled installs.

This expression is automatically updated within the [nixpkgs repo](https://github.com/NixOS/nixpkgs/blob/nixos-24.05/pkgs/by-name/ll/jarvis-cpp/package.nix#L164).

## Flox

On Mac and Linux, Flox can be used to install jarvis.cpp within a Flox environment via

```sh
flox install jarvis-cpp
```

Flox follows the nixpkgs build of jarvis.cpp.
