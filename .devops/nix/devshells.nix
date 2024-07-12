{
  perSystem =
    { config, lib, ... }:
    {
      devShells = lib.pipe (config.packages) [
        (lib.concatMapAttrs
        (name: package: {
          ${name} = package.passthru.shell or null;
        }))
        (lib.filterAttrs (name: value: value != null))
      ];
    };
}

