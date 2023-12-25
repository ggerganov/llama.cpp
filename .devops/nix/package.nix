{
  lib,
  config,
  stdenv,
  mkShell,
  cmake,
  ninja,
  pkg-config,
  git,
  python3,
  mpi,
  openblas, # TODO: Use the generic `blas` so users could switch betwen alternative implementations
  cudaPackages,
  darwin,
  rocmPackages,
  clblast,
  useBlas ? builtins.all (x: !x) [
    useCuda
    useMetalKit
    useOpenCL
    useRocm
  ],
  useCuda ? config.cudaSupport,
  useMetalKit ? stdenv.isAarch64 && stdenv.isDarwin && !useOpenCL,
  useOpenCL ? false,
  useRocm ? config.rocmSupport,
  llamaVersion ? "0.0.0", # Arbitrary version, substituted by the flake
}@inputs:

let
  inherit (lib)
    cmakeBool
    cmakeFeature
    optionals
    strings
    versionOlder
    ;

  # It's necessary to consistently use backendStdenv when building with CUDA support,
  # otherwise we get libstdc++ errors downstream.
  stdenv = throw "Use effectiveStdenv instead";
  effectiveStdenv = if useCuda then cudaPackages.backendStdenv else inputs.stdenv;

  suffices =
    lib.optionals useOpenCL [ "OpenCL" ]
    ++ lib.optionals useCuda [ "CUDA" ]
    ++ lib.optionals useRocm [ "ROCm" ]
    ++ lib.optionals useMetalKit [ "MetalKit" ]
    ++ lib.optionals useBlas [ "BLAS" ];

  pnameSuffix =
    strings.optionalString (suffices != [ ])
      "-${strings.concatMapStringsSep "-" strings.toLower suffices}";
  descriptionSuffix =
    strings.optionalString (suffices != [ ])
      ", accelerated with ${strings.concatStringsSep ", " suffices}";

  # TODO: package the Python in this repository in a Nix-like way.
  # It'd be nice to migrate to buildPythonPackage, as well as ensure this repo
  # is PEP 517-compatible, and ensure the correct .dist-info is generated.
  # https://peps.python.org/pep-0517/
  llama-python = python3.withPackages (
    ps: [
      ps.numpy
      ps.sentencepiece
    ]
  );

  # TODO(Green-Sky): find a better way to opt-into the heavy ml python runtime
  llama-python-extra = python3.withPackages (
    ps: [
      ps.numpy
      ps.sentencepiece
      ps.torchWithoutCuda
      ps.transformers
    ]
  );

  # apple_sdk is supposed to choose sane defaults, no need to handle isAarch64
  # separately
  darwinBuildInputs =
    with darwin.apple_sdk.frameworks;
    [
      Accelerate
      CoreVideo
      CoreGraphics
    ]
    ++ optionals useMetalKit [ MetalKit ];

  cudaBuildInputs = with cudaPackages; [
    cuda_cccl.dev # <nv/target>
    cuda_cudart
    libcublas
  ];

  rocmBuildInputs = with rocmPackages; [
    clr
    hipblas
    rocblas
  ];
in

effectiveStdenv.mkDerivation (
  finalAttrs: {
    pname = "llama-cpp${pnameSuffix}";
    version = llamaVersion;

    src = ../../.;

    postPatch = ''
      substituteInPlace ./ggml-metal.m \
        --replace '[bundle pathForResource:@"ggml-metal" ofType:@"metal"];' "@\"$out/bin/ggml-metal.metal\";"

      # TODO: Package up each Python script or service appropriately.
      # If we were to migrate to buildPythonPackage and prepare the `pyproject.toml`,
      # we could make those *.py into setuptools' entrypoints
      substituteInPlace ./*.py --replace "/usr/bin/env python" "${llama-python}/bin/python"
    '';

    nativeBuildInputs = [
      cmake
      ninja
      pkg-config
      git
    ] ++ optionals useCuda [ cudaPackages.cuda_nvcc ];

    buildInputs =
      [ mpi ]
      ++ optionals useOpenCL [ clblast ]
      ++ optionals useCuda cudaBuildInputs
      ++ optionals useRocm rocmBuildInputs
      ++ optionals effectiveStdenv.isDarwin darwinBuildInputs;

    cmakeFlags =
      [
        (cmakeBool "LLAMA_NATIVE" true)
        (cmakeBool "LLAMA_BUILD_SERVER" true)
        (cmakeBool "BUILD_SHARED_LIBS" true)
        (cmakeBool "CMAKE_SKIP_BUILD_RPATH" true)
        (cmakeBool "LLAMA_METAL" useMetalKit)
        (cmakeBool "LLAMA_BLAS" useBlas)
      ]
      ++ optionals useOpenCL [ (cmakeBool "LLAMA_CLBLAST" true) ]
      ++ optionals useCuda [ (cmakeBool "LLAMA_CUBLAS" true) ]
      ++ optionals useRocm [
        (cmakeBool "LLAMA_HIPBLAS" true)
        (cmakeFeature "CMAKE_C_COMPILER" "hipcc")
        (cmakeFeature "CMAKE_CXX_COMPILER" "hipcc")

        # Build all targets supported by rocBLAS. When updating search for TARGET_LIST_ROCM
        # in https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/CMakeLists.txt
        # and select the line that matches the current nixpkgs version of rocBLAS.
        # Should likely use `rocmPackages.clr.gpuTargets`.
        "-DAMDGPU_TARGETS=gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx940;gfx941;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102"
      ]
      ++ optionals useMetalKit [ (lib.cmakeFeature "CMAKE_C_FLAGS" "-D__ARM_FEATURE_DOTPROD=1") ]
      ++ optionals useBlas [ (lib.cmakeFeature "LLAMA_BLAS_VENDOR" "OpenBLAS") ];

    # TODO(SomeoneSerge): It's better to add proper install targets at the CMake level,
    # if they haven't been added yet.
    postInstall = ''
      mv $out/bin/main $out/bin/llama
      mv $out/bin/server $out/bin/llama-server
      mkdir -p $out/include
      cp $src/llama.h $out/include/
    '';

    # Define the shells here, but don't add in the inputsFrom to avoid recursion.
    passthru = {
      inherit
        useBlas
        useCuda
        useMetalKit
        useOpenCL
        useRocm
        ;

      shell = mkShell {
        name = "shell-${finalAttrs.finalPackage.name}";
        description = "contains numpy and sentencepiece";
        buildInputs = [ llama-python ];
        inputsFrom = [ finalAttrs.finalPackage ];
      };

      shell-extra = mkShell {
        name = "shell-extra-${finalAttrs.finalPackage.name}";
        description = "contains numpy, sentencepiece, torchWithoutCuda, and transformers";
        buildInputs = [ llama-python-extra ];
        inputsFrom = [ finalAttrs.finalPackage ];
      };
    };

    meta = {
      # Configurations we don't want even the CI to evaluate. Results in the
      # "unsupported platform" messages. This is mostly a no-op, because
      # cudaPackages would've refused to evaluate anyway.
      badPlatforms = optionals (useCuda || useOpenCL) lib.platforms.darwin;

      # Configurations that are known to result in build failures. Can be
      # overridden by importing Nixpkgs with `allowBroken = true`.
      broken = (useMetalKit && !effectiveStdenv.isDarwin);

      description = "Inference of LLaMA model in pure C/C++${descriptionSuffix}";
      homepage = "https://github.com/ggerganov/llama.cpp/";
      license = lib.licenses.mit;

      # Accommodates `nix run` and `lib.getExe`
      mainProgram = "llama";

      # These people might respond, on the best effort basis, if you ping them
      # in case of Nix-specific regressions or for reviewing Nix-specific PRs.
      # Consider adding yourself to this list if you want to ensure this flake
      # stays maintained and you're willing to invest your time. Do not add
      # other people without their consent. Consider removing people after
      # they've been unreachable for long periods of time.

      # Note that lib.maintainers is defined in Nixpkgs, but you may just add
      # an attrset following the same format as in
      # https://github.com/NixOS/nixpkgs/blob/f36a80e54da29775c78d7eff0e628c2b4e34d1d7/maintainers/maintainer-list.nix
      maintainers = with lib.maintainers; [
        philiptaron
        SomeoneSerge
      ];

      # Extend `badPlatforms` instead
      platforms = lib.platforms.all;
    };
  }
)
