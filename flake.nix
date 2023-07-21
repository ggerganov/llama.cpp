{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (pkgs.stdenv) isAarch32 isAarch64 isx86_32 isx86_64 isDarwin;
        osSpecific = with pkgs; [ openmpi ] ++
        (
          if isAarch64 && isDarwin then
            with pkgs.darwin.apple_sdk_11_0.frameworks; [
              Accelerate
              MetalKit
              MetalPerformanceShaders
              MetalPerformanceShadersGraph
            ]
          else if isAarch32 && isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else if isx86_32 || isx86_64 then
            with pkgs; [ mkl ]
          else
            with pkgs; [ openblas ]
        );
        pkgs = import nixpkgs { inherit system; };
        llama-python =
          pkgs.python310.withPackages (ps: with ps; [ numpy sentencepiece ]);
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "llama.cpp";
          src = ./.;
          postPatch = ''
            substituteInPlace ./ggml-metal.m \
              --replace '[bundle pathForResource:@"ggml-metal" ofType:@"metal"];' "@\"$out/bin/ggml-metal.metal\";"
          '';
          nativeBuildInputs = with pkgs; [ cmake pkgconfig ];
          buildInputs = osSpecific;
          cmakeFlags = [ "-DLLAMA_BUILD_SERVER=ON" "-DLLAMA_MPI=ON" "-DBUILD_SHARED_LIBS=ON" "-DCMAKE_SKIP_BUILD_RPATH=ON" ]
            ++ (if isAarch64 && isDarwin then [
              "-DCMAKE_C_FLAGS=-D__ARM_FEATURE_DOTPROD=1"
              "-DLLAMA_METAL=ON"
            ] else if isx86_32 || isx86_64 then [
              "-DLLAMA_BLAS=ON"
              "-DLLAMA_BLAS_VENDOR=Intel10_lp64"
            ] else [
              "-DLLAMA_BLAS=ON"
              "-DLLAMA_BLAS_VENDOR=OpenBLAS"
          ]);
          installPhase = ''
            runHook preInstall

            install -D bin/* -t $out/bin
            install -Dm644 lib*.so -t $out/lib
            mv $out/bin/main $out/bin/llama
            mv $out/bin/server $out/bin/llama-server

            echo "#!${llama-python}/bin/python" > $out/bin/convert.py
            cat ${./convert.py} >> $out/bin/convert.py
            chmod +x $out/bin/convert.py

            runHook postInstall
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
