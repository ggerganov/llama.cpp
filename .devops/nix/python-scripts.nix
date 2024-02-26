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
  src = ../../.;
  version = "0.0.0";
  pyproject = true;
  nativeBuildInputs = [ poetry-core ];
  projectDir = ../../.;
  propagatedBuildInputs = llama-python-deps;

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
