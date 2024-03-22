{
  lib,
  glibc,
  config,
  stdenv,
  mkShell,
  cmake,
  ninja,
  pkg-config,
  git,
  python3,
  mpi,
  openblas, # TODO: Use the generic `blas` so users could switch between alternative implementations
  cudaPackages,
  darwin,
  rocmPackages,
  vulkan-headers,
  vulkan-loader,
  clblast,
  useBlas ? builtins.all (x: !x) [
    useCuda
    useMetalKit
    useOpenCL
    useRocm
    useVulkan
  ],
  useCuda ? config.cudaSupport,
  useMetalKit ? stdenv.isAarch64 && stdenv.isDarwin && !useOpenCL,
  useMpi ? false, # Increases the runtime closure size by ~700M
  useOpenCL ? false,
  useRocm ? config.rocmSupport,
  useVulkan ? false,
  llamaVersion ? "0.0.0", # Arbitrary version, substituted by the flake

  # It's necessary to consistently use backendStdenv when building with CUDA support,
  # otherwise we get libstdc++ errors downstream.
  effectiveStdenv ? if useCuda then cudaPackages.backendStdenv else stdenv,
  enableStatic ? effectiveStdenv.hostPlatform.isStatic
}@inputs:

let
  inherit (lib)
    cmakeBool
    cmakeFeature
    optionals
    strings
    versionOlder
    ;

  stdenv = throw "Use effectiveStdenv instead";

  suffices =
    lib.optionals useBlas [ "BLAS" ]
    ++ lib.optionals useCuda [ "CUDA" ]
    ++ lib.optionals useMetalKit [ "MetalKit" ]
    ++ lib.optionals useMpi [ "MPI" ]
    ++ lib.optionals useOpenCL [ "OpenCL" ]
    ++ lib.optionals useRocm [ "ROCm" ]
    ++ lib.optionals useVulkan [ "Vulkan" ];

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
      ps.tiktoken
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

    # A temporary hack for reducing the closure size, remove once cudaPackages
    # have stopped using lndir: https://github.com/NixOS/nixpkgs/issues/271792
    cuda_cudart.dev
    cuda_cudart.lib
    cuda_cudart.static
    libcublas.dev
    libcublas.lib
    libcublas.static
  ];

  rocmBuildInputs = with rocmPackages; [
    clr
    hipblas
    rocblas
  ];

  vulkanBuildInputs = [
    vulkan-headers
    vulkan-loader
  ];
in

effectiveStdenv.mkDerivation (
  finalAttrs: {
    pname = "llama-cpp${pnameSuffix}";
    version = llamaVersion;

    # Note: none of the files discarded here are visible in the sandbox or
    # affect the output hash. This also means they can be modified without
    # triggering a rebuild.
    src = lib.cleanSourceWith {
      filter =
        name: type:
        let
          noneOf = builtins.all (x: !x);
          baseName = baseNameOf name;
        in
        noneOf [
          (lib.hasSuffix ".nix" name) # Ignore *.nix files when computing outPaths
          (lib.hasSuffix ".md" name) # Ignore *.md changes whe computing outPaths
          (lib.hasPrefix "." baseName) # Skip hidden files and directories
          (baseName == "flake.lock")
        ];
      src = lib.cleanSource ../../.;
    };

    postPatch = ''
      substituteInPlace ./ggml-metal.m \
        --replace '[bundle pathForResource:@"ggml-metal" ofType:@"metal"];' "@\"$out/bin/ggml-metal.metal\";"

      # TODO: Package up each Python script or service appropriately.
      # If we were to migrate to buildPythonPackage and prepare the `pyproject.toml`,
      # we could make those *.py into setuptools' entrypoints
      substituteInPlace ./*.py --replace "/usr/bin/env python" "${llama-python}/bin/python"
    '';

    nativeBuildInputs =
      [
        cmake
        ninja
        pkg-config
        git
      ]
      ++ optionals useCuda [
        cudaPackages.cuda_nvcc

        # TODO: Replace with autoAddDriverRunpath
        # once https://github.com/NixOS/nixpkgs/pull/275241 has been merged
        cudaPackages.autoAddOpenGLRunpathHook
      ]
      ++ optionals (effectiveStdenv.hostPlatform.isGnu && enableStatic) [
        glibc.static
      ];

    buildInputs =
      optionals effectiveStdenv.isDarwin darwinBuildInputs
      ++ optionals useCuda cudaBuildInputs
      ++ optionals useMpi [ mpi ]
      ++ optionals useOpenCL [ clblast ]
      ++ optionals useRocm rocmBuildInputs
      ++ optionals useVulkan vulkanBuildInputs;

    cmakeFlags =
      [
        (cmakeBool "LLAMA_NATIVE" false)
        (cmakeBool "LLAMA_BUILD_SERVER" true)
        (cmakeBool "BUILD_SHARED_LIBS" (!enableStatic))
        (cmakeBool "CMAKE_SKIP_BUILD_RPATH" true)
        (cmakeBool "LLAMA_BLAS" useBlas)
        (cmakeBool "LLAMA_CLBLAST" useOpenCL)
        (cmakeBool "LLAMA_CUBLAS" useCuda)
        (cmakeBool "LLAMA_HIPBLAS" useRocm)
        (cmakeBool "LLAMA_METAL" useMetalKit)
        (cmakeBool "LLAMA_MPI" useMpi)
        (cmakeBool "LLAMA_VULKAN" useVulkan)
        (cmakeBool "LLAMA_STATIC" enableStatic)
      ]
      ++ optionals useCuda [
        (
          with cudaPackages.flags;
          cmakeFeature "CMAKE_CUDA_ARCHITECTURES" (
            builtins.concatStringsSep ";" (map dropDot cudaCapabilities)
          )
        )
      ]
      ++ optionals useRocm [
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
        useMpi
        useOpenCL
        useRocm
        useVulkan
        ;

      shell = mkShell {
        name = "shell-${finalAttrs.finalPackage.name}";
        description = "contains numpy and sentencepiece";
        buildInputs = [ llama-python ];
        inputsFrom = [ finalAttrs.finalPackage ];
        shellHook = ''
          addToSearchPath "LD_LIBRARY_PATH" "${lib.getLib effectiveStdenv.cc.cc}/lib"
        '';
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
