{ concatMapAttrs, packages }:

concatMapAttrs
  (name: package: {
    ${name} = package.passthru.shell;
    ${name + "-extra"} = package.passthru.shell-extra;
  })
  packages
