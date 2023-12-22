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
      # These define the various ways to build the llama.cpp project.
      # Integrate them into your flake.nix configuration by adding this overlay to nixpkgs.overlays.
      overlays.default = import ./.devops/nix/overlay.nix;

      # These use the package definition from `./.devops/nix/package.nix`.
      # There's one per backend that llama-cpp uses. Add more as needed!
      packages = eachSystem (
        system:
        let
          defaultConfig = {
            inherit system;
            overlays = [ self.overlays.default ];
          };
          pkgs = import nixpkgs defaultConfig;

          # Let's not make a big deal about getting the CUDA bits.
          cudaConfig = defaultConfig // {
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
          pkgsCuda = import nixpkgs cudaConfig;

          # Let's make sure to turn on ROCm support across the whole package ecosystem.
          rocmConfig = defaultConfig // {
            config.rocmSupport = true;
          };
          pkgsRocm = import nixpkgs rocmConfig;
        in
        {
          default = pkgs.llama-cpp;
          opencl = pkgs.llama-cpp.override { useOpenCL = true; };
          cuda = pkgsCuda.llama-cpp;
          rocm = pkgsRocm.llama-cpp;
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
