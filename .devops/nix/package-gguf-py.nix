{
  lib,
  llamaVersion,
  numpy,
  tqdm,
  sentencepiece,
  poetry-core,
  buildPythonPackage,
  pytestCheckHook,
}@inputs:

buildPythonPackage {
  pname = "gguf";
  version = llamaVersion;
  pyproject = true;
  nativeBuildInputs = [ poetry-core ];
  propagatedBuildInputs = [ numpy tqdm sentencepiece ];
  src = lib.cleanSource ../../gguf-py;
  pythonImportsCheck = [
    "numpy"
    "gguf"
  ];
  nativeCheckInputs = [ pytestCheckHook ];
  doCheck = true;
  meta = with lib; {
    description = "Python package for writing binary files in the GGUF format";
    license = licenses.mit;
    maintainers = [ maintainers.ditsuke ];
  };
}
