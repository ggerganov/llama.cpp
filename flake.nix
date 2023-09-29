{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        name = "llama.cpp";
        src = ./.;
        meta.mainProgram = "llama";
        inherit (pkgs.stdenv) isAarch32 isAarch64 isDarwin;
        buildInputs = with pkgs; [ openmpi ];
        osSpecific = with pkgs; buildInputs ++
        (
          if isAarch64 && isDarwin then
            with pkgs.darwin.apple_sdk_11_0.frameworks; [
              Accelerate
              MetalKit
            ]
          else if isAarch32 && isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else if isDarwin then
            with pkgs.darwin.apple_sdk.frameworks; [
              Accelerate
              CoreGraphics
              CoreVideo
            ]
          else
            with pkgs; [ openblas ]
        );
        pkgs = import nixpkgs { inherit system; };
        nativeBuildInputs = with pkgs; [ cmake ninja pkg-config ];
        cudatoolkit_joined = with pkgs; symlinkJoin {
          # HACK(Green-Sky): nix currently has issues with cmake findcudatoolkit
          # see https://github.com/NixOS/nixpkgs/issues/224291
          # copied from jaxlib
          name = "${cudaPackages.cudatoolkit.name}-merged";
          paths = [
            cudaPackages.cudatoolkit.lib
            cudaPackages.cudatoolkit.out
          ] ++ lib.optionals (lib.versionOlder cudaPackages.cudatoolkit.version "11") [
            # for some reason some of the required libs are in the targets/x86_64-linux
            # directory; not sure why but this works around it
            "${cudaPackages.cudatoolkit}/targets/${system}"
          ];
        };
        llama-python =
          pkgs.python3.withPackages (ps: with ps; [ numpy sentencepiece ]);
        postPatch = ''
          substituteInPlace ./ggml-metal.m \
            --replace '[bundle pathForResource:@"ggml-metal" ofType:@"metal"];' "@\"$out/bin/ggml-metal.metal\";"
          substituteInPlace ./*.py --replace '/usr/bin/env python' '${llama-python}/bin/python'
        '';
        postInstall = ''
          mv $out/bin/main $out/bin/llama
          mv $out/bin/server $out/bin/llama-server
          mkdir -p $out/include
          cp ${src}/llama.h $out/include/
        '';
        cmakeFlags = [ "-DLLAMA_BUILD_SERVER=ON" "-DLLAMA_MPI=ON" "-DBUILD_SHARED_LIBS=ON" "-DCMAKE_SKIP_BUILD_RPATH=ON" ];
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          inherit name src meta postPatch nativeBuildInputs postInstall;
          buildInputs = osSpecific;
          cmakeFlags = cmakeFlags
            ++ (if isAarch64 && isDarwin then [
            "-DCMAKE_C_FLAGS=-D__ARM_FEATURE_DOTPROD=1"
            "-DLLAMA_METAL=ON"
          ] else [
            "-DLLAMA_BLAS=ON"
            "-DLLAMA_BLAS_VENDOR=OpenBLAS"
          ]);
        };
        packages.opencl = pkgs.stdenv.mkDerivation {
          inherit name src meta postPatch nativeBuildInputs postInstall;
          buildInputs = with pkgs; buildInputs ++ [ clblast ];
          cmakeFlags = cmakeFlags ++ [
            "-DLLAMA_CLBLAST=ON"
          ];
        };
        packages.cuda = pkgs.stdenv.mkDerivation {
          inherit name src meta postPatch nativeBuildInputs postInstall;
          buildInputs = with pkgs; buildInputs ++ [ cudatoolkit_joined ];
          cmakeFlags = cmakeFlags ++ [
            "-DLLAMA_CUBLAS=ON"
          ];
        };
        packages.rocm = pkgs.stdenv.mkDerivation {
          inherit name src meta postPatch nativeBuildInputs postInstall;
          buildInputs = with pkgs; buildInputs ++ [ hip hipblas rocblas ];
          cmakeFlags = cmakeFlags ++ [
            "-DLLAMA_HIPBLAS=1"
            "-DCMAKE_C_COMPILER=hipcc"
            "-DCMAKE_CXX_COMPILER=hipcc"
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
          ];
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
        apps.quantize = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/quantize";
        };
        apps.train-text-from-scratch = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/train-text-from-scratch";
        };
        apps.default = self.apps.${system}.llama;
        devShells.default = pkgs.mkShell {
          buildInputs = [ llama-python ];
          packages = nativeBuildInputs ++ osSpecific;
        };
      });
}
