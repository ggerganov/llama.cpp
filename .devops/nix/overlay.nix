final: prev:

let
  inherit (final.stdenv) isAarch64 isDarwin;

  darwinSpecific =
    if isAarch64 then
      { inherit (final.darwin.apple_sdk_11_0.frameworks) Accelerate MetalKit; }
    else
      { inherit (final.darwin.apple_sdk.frameworks) Accelerate CoreGraphics CoreVideo; };

  osSpecific = if isDarwin then darwinSpecific else { };
in

{
  llama-cpp = final.callPackage ./package.nix osSpecific;
}
