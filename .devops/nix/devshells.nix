{
  perSystem =
    { config, lib, ... }:
    {
      devShells = lib.concatMapAttrs (name: package: {
        ${name} = package.passthru.shell;
      }) config.packages;
    };
}
