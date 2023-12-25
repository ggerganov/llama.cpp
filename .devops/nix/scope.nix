{ lib, newScope }:

lib.makeScope newScope (self: { llama-cpp = self.callPackage ./package.nix { }; })
