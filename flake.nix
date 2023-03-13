{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        llama-python = with pkgs; python310.withPackages (ps: with python310Packages; [
          torch
          numpy
          sentencepiece
        ]);
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "llama.cpp";
          buildInputs = with pkgs; lib.optionals stdenv.isDarwin [
            darwin.apple_sdk.frameworks.Accelerate
          ];
          src = ./.;
          installPhase = ''
            mkdir -p $out/bin
            mv main $out/bin/llama-cpp
            mv quantize $out/bin/llama-cpp-quantize
            echo "#!${llama-python}/bin/python" > $out/bin/convert-pth-to-ggml
            cat convert-pth-to-ggml.py >> $out/bin/convert-pth-to-ggml
            chmod +x $out/bin/convert-pth-to-ggml
          '';
        };
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            llama-python
          ] ++ lib.optionals stdenv.isDarwin [
            darwin.apple_sdk.frameworks.Accelerate
          ];
        };
      }
    );
}
