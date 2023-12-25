{
  lib,
  newScope,
  llamaVersion ? "0.0.0",
}:

lib.makeScope newScope (
  self: {
    inherit llamaVersion;
    llama-cpp = self.callPackage ./package.nix { };
  }
)
