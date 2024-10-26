{ inputs, ... }:
{
  perSystem =
    {
      config,
      system,
      lib,
      pkgsCuda,
      ...
    }:
    {
      legacyPackages =
        let
          caps.jarvisPackagesXavier = "7.2";
          caps.jarvisPackagesOrin = "8.7";
          caps.jarvisPackagesTX2 = "6.2";
          caps.jarvisPackagesNano = "5.3";

          pkgsFor =
            cap:
            import inputs.nixpkgs {
              inherit system;
              config = {
                cudaSupport = true;
                cudaCapabilities = [ cap ];
                cudaEnableForwardCompat = false;
                inherit (pkgsCuda.config) allowUnfreePredicate;
              };
            };
        in
        builtins.mapAttrs (name: cap: (pkgsFor cap).callPackage ./scope.nix { }) caps;

      packages = lib.optionalAttrs (system == "aarch64-linux") {
        jetson-xavier = config.legacyPackages.jarvisPackagesXavier.jarvis-cpp;
        jetson-orin = config.legacyPackages.jarvisPackagesOrin.jarvis-cpp;
        jetson-nano = config.legacyPackages.jarvisPackagesNano.jarvis-cpp;
      };
    };
}
