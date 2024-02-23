{
  lib,
  llamaVersion,
  numpy,
  poetry-core,
  buildPythonPackage,
}@inputs:

buildPythonPackage {
  pname = "gguf";
  version = llamaVersion;
  pyproject = true;
  nativeBuildInputs = [ poetry-core ];
  propagatedBuildInputs = [ numpy ];
  src = lib.cleanSource ../../gguf-py;
  doCheck = false;
  meta = with lib; {
    description = "Python package for writing binary files in the GGUF format";
    license = licenses.mit;
    maintainers = [ maintainers.ditsuke ];
  };
}
