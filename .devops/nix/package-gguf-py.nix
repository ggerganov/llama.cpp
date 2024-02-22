{
  lib,
  llamaVersion,
  python3,
}@inputs:

python3.pkgs.buildPythonPackage rec {
  pname = "gguf";
  version = llamaVersion;
  pyproject = true;
  nativeBuildInputs = with python3.pkgs; [ poetry-core ];
  propagatedBuildInputs = with python3.pkgs; [ numpy ];
  src = lib.cleanSource ../../gguf-py;
  doCheck = false;
  meta = with lib; {
    description = "Python package for writing binary files in the GGUF format";
    license = licenses.mit;
    maintainers = [ maintainers.ditsuke ];
  };
}
