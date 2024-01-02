{ inputs, ... }:
{
  # The _module.args definitions are passed on to modules as arguments. E.g.
  # the module `{ pkgs ... }: { /* config */ }` implicitly uses
  # `_module.args.pkgs` (defined in this case by flake-parts).
  perSystem =
    { system, ... }:
    {
      _module.args = {
        pkgsCuda = import inputs.nixpkgs {
          inherit system;
          # Ensure dependencies use CUDA consistently (e.g. that openmpi, ucc,
          # and ucx are built with CUDA support)
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
        pkgsRocm = import inputs.nixpkgs {
          inherit system;
          config.rocmSupport = true;
        };
      };
    };
}
