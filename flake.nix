{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:

    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin" # x86_64-darwin isn't tested (and likely isn't relevant)
        "x86_64-linux"
      ];
      eachSystem = f: nixpkgs.lib.genAttrs systems (system: f system);
    in

    {
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
      overlays.default = (final: prev: { llamaPackages = final.callPackage .devops/nix/scope.nix { }; });

      # These use the package definition from `./.devops/nix/package.nix`.
      # There's one per backend that llama-cpp uses. Add more as needed!
      packages = eachSystem (
        system:
        let
          # Avoid re-evaluation for the nixpkgs instance,
          # cf. https://zimbatm.com/notes/1000-instances-of-nixpkgs
          pkgs = nixpkgs.legacyPackages.${system};

          # Ensure dependencies use CUDA consistently (e.g. that openmpi, ucc,
          # and ucx are built with CUDA support)
          pkgsCuda = import nixpkgs {
            inherit system;

            config.cudaSupport = true;
            config.allowUnfreePredicate =
              p:
              builtins.all
                (
                  license:
                  license.free
                  || builtins.elem license.shortName [
                    "CUDA EULA"
                    "cuDNN EULA"
                  ]
                )
                (p.meta.licenses or [ p.meta.license ]);
          };

          # Ensure dependencies use ROCm consistently
          pkgsRocm = import nixpkgs {
            inherit system;
            config.rocmSupport = true;
          };
        in
        {
          default = (pkgs.callPackage .devops/nix/scope.nix { }).llama-cpp;
          opencl = self.packages.${system}.default.override { useOpenCL = true; };
          cuda = (pkgsCuda.callPackage .devops/nix/scope.nix { }).llama-cpp;
          rocm = (pkgsRocm.callPackage .devops/nix/scope.nix { }).llama-cpp;
        }
      );

      # These use the definition of llama-cpp from `./.devops/nix/package.nix`
      # and expose various binaries as apps with `nix run .#app-name`.
      # Note that none of these apps use anything other than the default backend.
      apps = eachSystem (
        system:
        import ./.devops/nix/apps.nix {
          package = self.packages.${system}.default;
          binaries = [
            "llama"
            "llama-embedding"
            "llama-server"
            "quantize"
            "train-text-from-scratch"
          ];
        }
      );

      # These expose a build environment for either a "default" or an "extra" set of dependencies.
      devShells = eachSystem (
        system:
        import ./.devops/nix/devshells.nix {
          concatMapAttrs = nixpkgs.lib.concatMapAttrs;
          packages = self.packages.${system};
        }
      );
    };
}
