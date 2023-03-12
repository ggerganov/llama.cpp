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
        python = with pkgs; python310.withPackages (ps: with python310Packages; [
          torch
          numpy
          sentencepiece
        ]);
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "llama.cpp";
          buildInputs = [
            pkgs.darwin.apple_sdk.frameworks.Accelerate
          ];
          src = ./.;
          installPhase = ''
            mkdir -p $out/bin
            mv main $out/bin/llama-cpp
            mv quantize $out/bin/llama-cpp-quantize
            echo "#!${python}/bin/python" > $out/bin/convert-pth-to-ggml
            cat convert-pth-to-ggml.py >> $out/bin/convert-pth-to-ggml
            chmod +x $out/bin/convert-pth-to-ggml
          '';
        };
      }
    );
}
