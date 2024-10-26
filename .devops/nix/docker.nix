{
  lib,
  dockerTools,
  buildEnv,
  jarvis-cpp,
  interactive ? true,
  coreutils,
}:

# A tar that can be fed into `docker load`:
#
# $ nix build .#jarvisPackages.docker
# $ docker load < result

# For details and variations cf.
# - https://nixos.org/manual/nixpkgs/unstable/#ssec-pkgs-dockerTools-buildLayeredImage
# - https://discourse.nixos.org/t/a-faster-dockertools-buildimage-prototype/16922
# - https://nixery.dev/

# Approximate (compressed) sizes, at the time of writing, are:
#
# .#jarvisPackages.docker: 125M;
# .#jarvisPackagesCuda.docker: 537M;
# .#legacyPackages.aarch64-linux.jarvisPackagesXavier.docker: 415M.

dockerTools.buildLayeredImage {
  name = jarvis-cpp.pname;
  tag = "latest";

  contents =
    [ jarvis-cpp ]
    ++ lib.optionals interactive [
      coreutils
      dockerTools.binSh
      dockerTools.caCertificates
    ];
}
