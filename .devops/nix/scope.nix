{
  lib,
  newScope,
  python3,
  llamaVersion ? "0.0.0",
}:

let
  pythonPackages = python3.pkgs;
  buildPythonPackage = pythonPackages.buildPythonPackage;
  numpy = pythonPackages.numpy;
  poetry-core = pythonPackages.poetry-core;
  pytestCheckHook = pythonPackages.pytestCheckHook;
in

# We're using `makeScope` instead of just writing out an attrset
# because it allows users to apply overlays later using `overrideScope'`.
# Cf. https://noogle.dev/f/lib/makeScope

lib.makeScope newScope (self: {
  inherit llamaVersion;
  pp = python3.pkgs;
  gguf-py = self.callPackage ./package-gguf-py.nix {
    inherit
      buildPythonPackage
      numpy
      poetry-core
      pytestCheckHook
      ;
  };
  llama-cpp = self.callPackage ./package.nix { };
  docker = self.callPackage ./docker.nix { };
  docker-min = self.callPackage ./docker.nix { interactive = false; };
  sif = self.callPackage ./sif.nix { };
})
