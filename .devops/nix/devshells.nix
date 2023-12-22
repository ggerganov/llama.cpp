{ concatMapAttrs, packages }:

concatMapAttrs
  (name: package: {
    ${name} = package.passthru.shell.overrideAttrs (prevAttrs: { inputsFrom = [ package ]; });
    ${name + "-extra"} = package.passthru.shell-extra.overrideAttrs (
      prevAttrs: { inputsFrom = [ package ]; }
    );
  })
  packages
