{
  perSystem =
    { config, lib, ... }:
    {
      devShells =
        lib.concatMapAttrs
          (name: package: {
            ${name} = package.passthru.shell;
            ${name + "-extra"} = package.passthru.shell-extra;
          })
          config.packages;
    };
}
