self: super:

{
  cosmo = super.stdenv.mkDerivation rec {
    pname = "cosmo";
    version = "1.0.0";

    src = super.fetchurl {
      url = "https://cosmonic.sh/install.sh";
      sha256 = "1961f948b184b31a820a68c01388d7c8e2e21c47285b63407de07a774b60b7f8";
    };

    buildInputs = [ super.curl super.cacert super.which ];

    phases = [ "installPhase" ];

    installPhase = ''
    mkdir -p $out/bin
    cp $src $out/bin/cosmo
    chmod +x $out/bin/cosmo

    # Replace 'curl' with the absolute path to the curl binary from the Nix store
    sed -i 's|curl|${super.curl}/bin/curl|g' $out/bin/cosmo
    # Replace 'which' with the absolute path to the which binary from the Nix store
    sed -i 's|which|${super.which}/bin/which|g' $out/bin/cosmo

    # Set the installation directory to $out/bin
    export COSMO_INSTALL_DIR=$out/bin

    # Replace the original installation directory with $out/bin
    sed -i 's|~/.cosmo/bin|$out/bin|g' $out/bin/cosmo

    # Run the modified installation script
    bash $out/bin/cosmo
    '';
  };
}
