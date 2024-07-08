{
  lib,
  stdenv,
  buildPythonPackage,
  poetry-core,
  breakpointHook,
  mkShell,
  python3Packages,
  gguf-py,
}@inputs:

let
  llama-python-deps = with python3Packages; [
    numpy
    sentencepiece
    transformers
    protobuf
    torchWithoutCuda
    gguf-py
    tqdm
  ];
in

buildPythonPackage ({
  pname = "llama-scripts";
  version = "0.0.0";
  pyproject = true;

  # NOTE: The files filtered out here are not visible in the build sandbox, neither
  # do they affect the output hash. They can be modified without triggering a rebuild.
  src = lib.cleanSourceWith {
    filter =
      name: type:
      let
        any = builtins.any (x: x);
        baseName = builtins.baseNameOf name;
      in
      any [
        (lib.hasSuffix ".py" name)
        (baseName == "README.md")
        (baseName == "pyproject.toml")
      ];
    src = lib.cleanSource ../../.;
  };
  nativeBuildInputs = [ poetry-core ];
  dependencies = llama-python-deps;

  passthru = {
    shell = mkShell {
      name = "shell-python-scripts";
      description = "contains numpy and sentencepiece";
      buildInputs = llama-python-deps;
      shellHook = ''
        addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib stdenv.cc.cc}/lib"
      '';
    };
  };
})
