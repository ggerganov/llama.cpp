{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        llama-python = pkgs.python310.withPackages (ps: with ps; [
          numpy
          sentencepiece
        ]);
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "llama.cpp";
          src = ./.;
          nativeBuildInputs = with pkgs; [ cmake ];
          buildInputs = with pkgs; lib.optionals stdenv.isDarwin [
            darwin.apple_sdk.frameworks.Accelerate
          ];
          cmakeFlags = with pkgs; lib.optionals (system == "aarch64-darwin") [
            "-DCMAKE_C_FLAGS=-D__ARM_FEATURE_DOTPROD=1"
          ];
          installPhase = ''
            mkdir -p $out/bin
            mv bin/* $out/bin/
            mv $out/bin/main $out/bin/llama

            echo "#!${llama-python}/bin/python" > $out/bin/convert.py
            cat ${./convert.py} >> $out/bin/convert.py
            chmod +x $out/bin/convert.py
          '';
          meta.mainProgram = "llama";
        };
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            llama-python
          ] ++ lib.optionals stdenv.isDarwin [
            darwin.apple_sdk.frameworks.Accelerate
          ];
        };
      }
    );
}
