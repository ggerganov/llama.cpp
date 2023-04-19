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
          buildInputs = with pkgs; lib.optionals stdenv.isDarwin [
            darwin.apple_sdk.frameworks.Accelerate
          ];
          makeFlags = with pkgs; lib.optionals (system == "aarch64-darwin") [
            "CFLAGS=-D__ARM_FEATURE_DOTPROD=1"
          ];
          buildPhase = ''
            make main quantize quantize-stats perplexity embedding vdot libllama.so
          '';
          installPhase = ''
            mkdir -p $out/lib/
            cp libllama.so $out/lib/

            mkdir -p $out/bin/
            mv main $out/bin/llama
            for exe in quantize quantize-stats perplexity embedding vdot; do
              mv $exe $out/bin/
            done

            echo "#!${llama-python}/bin/python" > $out/bin/convert-pth-to-ggml
            cat ${./convert-pth-to-ggml.py} >> $out/bin/convert-pth-to-ggml
            chmod +x $out/bin/convert-pth-to-ggml
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
