self: super:

{
  vespa-cli = super.stdenv.mkDerivation rec {
    pname = "vespa-cli";
    version = "8.162.29";

    src = super.fetchurl {
      url = "https://github.com/vespa-engine/vespa/releases/download/v${version}/vespa-cli_${version}_darwin_arm64.tar.gz";
      sha256 = "05cdabe044ac34a74eca66ade3db49b7a1db29c691e7f8bbd1e71ab991548688";
    };

    buildInputs = [];

    nativeBuildInputs = [ super.makeWrapper ];

    dontBuild = true;

    unpackPhase = ''
      tar -xzf $src
    '';

    installPhase = ''
      mkdir -p $out/bin
      cp vespa-cli_${version}_darwin_arm64/bin/vespa $out/bin
      chmod +x $out/bin/vespa
      # Wrap the vespa binary to remove quarantine attribute
    '';
  };
}
