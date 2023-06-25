{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (pkgs.stdenv) isAarch64 isDarwin;
        inherit (pkgs.lib) optionals;
        isM1 = isAarch64 && isDarwin;
        osSpecific = if isM1 then
          with pkgs.darwin.apple_sdk_11_0.frameworks; [
            Accelerate
            MetalKit
            MetalPerformanceShaders
            MetalPerformanceShadersGraph
          ]
        else if isDarwin then
          with pkgs.darwin.apple_sdk.frameworks; [
            Accelerate
            CoreGraphics
            CoreVideo
          ]
        else
          [ ];
        pkgs = import nixpkgs { inherit system; };
        llama-python =
          pkgs.python310.withPackages (ps: with ps; [ numpy sentencepiece ]);
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "llama.cpp";
          src = ./.;
          postPatch = if isM1 then ''
            substituteInPlace ./ggml-metal.m \
              --replace '[bundle pathForResource:@"ggml-metal" ofType:@"metal"];' "@\"$out/bin/ggml-metal.metal\";"
          '' else
            "";
          nativeBuildInputs = with pkgs; [ cmake ];
          buildInputs = osSpecific;
          cmakeFlags = [ "-DLLAMA_BUILD_SERVER=ON" ] ++ (optionals isM1 [
            "-DCMAKE_C_FLAGS=-D__ARM_FEATURE_DOTPROD=1"
            "-DLLAMA_METAL=ON"
          ]);
          installPhase = ''
            mkdir -p $out/bin
            mv bin/* $out/bin/
            mv $out/bin/main $out/bin/llama
            mv $out/bin/server $out/bin/llama-server

            echo "#!${llama-python}/bin/python" > $out/bin/convert.py
            cat ${./convert.py} >> $out/bin/convert.py
            chmod +x $out/bin/convert.py
          '';
          meta.mainProgram = "llama";
        };
        apps.llama-server = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/llama-server";
        };
        apps.llama-embedding = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/embedding";
        };
        apps.llama = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/llama";
        };
        apps.default = self.apps.${system}.llama;
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [ cmake llama-python ] ++ osSpecific;
        };
      });
}
