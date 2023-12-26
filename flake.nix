{
  description = "Port of Facebook's LLaMA model in C/C++";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  # For inspection, use `nix flake show github:ggerganov/llama.cpp` or the nix repl:
  #
  # ```bash
  # ❯ nix repl
  # nix-repl> :lf github:ggerganov/llama.cpp
  # Added 13 variables.
  # nix-repl> outputs.apps.x86_64-linux.quantize
  # { program = "/nix/store/00000000000000000000000000000000-llama.cpp/bin/quantize"; type = "app"; }
  # ```
  outputs =
    { self, flake-parts, ... }@inputs:
    let
      # We could include the git revisions in the package names but those would
      # needlessly trigger rebuilds:
      # llamaVersion = self.dirtyShortRev or self.shortRev;

      # Nix already uses cryptographic hashes for versioning, so we'll just fix
      # the fake semver for now:
      llamaVersion = "0.0.0";
    in
    flake-parts.lib.mkFlake { inherit inputs; }

      {

        imports = [
          .devops/nix/nixpkgs-instances.nix
          .devops/nix/apps.nix
          .devops/nix/devshells.nix
          .devops/nix/jetson-support.nix
        ];

        # An overlay can be used to have a more granular control over llama-cpp's
        # dependencies and configuration, than that offered by the `.override`
        # mechanism. Cf. https://nixos.org/manual/nixpkgs/stable/#chap-overlays.
        #
        # E.g. in a flake:
        # ```
        # { nixpkgs, llama-cpp, ... }:
        # let pkgs = import nixpkgs {
        #     overlays = [ (llama-cpp.overlays.default) ];
        #     system = "aarch64-linux";
        #     config.allowUnfree = true;
        #     config.cudaSupport = true;
        #     config.cudaCapabilities = [ "7.2" ];
        #     config.cudaEnableForwardCompat = false;
        # }; in {
        #     packages.aarch64-linux.llamaJetsonXavier = pkgs.llamaPackages.llama-cpp;
        # }
        # ```
        #
        # Cf. https://nixos.org/manual/nix/unstable/command-ref/new-cli/nix3-flake.html?highlight=flake#flake-format
        flake.overlays.default =
          (final: prev: {
            llamaPackages = final.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            inherit (final.llamaPackages) llama-cpp;
          });

        systems = [
          "aarch64-darwin"
          "aarch64-linux"
          "x86_64-darwin" # x86_64-darwin isn't tested (and likely isn't relevant)
          "x86_64-linux"
        ];

        perSystem =
          {
            config,
            lib,
            pkgs,
            pkgsCuda,
            pkgsRocm,
            ...
          }:
          {
            # We don't use the overlay here so as to avoid making too many instances of nixpkgs,
            # cf. https://zimbatm.com/notes/1000-instances-of-nixpkgs
            packages =
              {
                default = (pkgs.callPackage .devops/nix/scope.nix { inherit llamaVersion; }).llama-cpp;
              }
              // lib.optionalAttrs pkgs.stdenv.isLinux {
                opencl = config.packages.default.override { useOpenCL = true; };
                cuda = (pkgsCuda.callPackage .devops/nix/scope.nix { inherit llamaVersion; }).llama-cpp;
                rocm = (pkgsRocm.callPackage .devops/nix/scope.nix { inherit llamaVersion; }).llama-cpp;

                mpi-cpu = config.packages.default.override { useMpi = true; };
                mpi-cuda = config.packages.default.override { useMpi = true; };
              };
          };
      };
}
