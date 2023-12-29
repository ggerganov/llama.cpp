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
    lib.optionalAttrs (system == "aarch64-linux") {
      packages =
        let
          caps.jetson-xavier = "7.2";
          caps.jetson-orin = "8.7";
          caps.jetson-nano = "5.3";

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
        builtins.mapAttrs (name: cap: ((pkgsFor cap).callPackage ./scope.nix { }).llama-cpp) caps;
    };
}
