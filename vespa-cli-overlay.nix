self: super:

{
  vespa-cli = super.stdenv.mkDerivation rec {
    pname = "vespa-cli";
    version = "8.167.17";

    src = super.fetchurl {
      url = "https://github.com/vespa-engine/vespa/releases/download/v${version}/vespa-cli_${version}_darwin_arm64.tar.gz";
      sha256 = "df8ec80d0f44bc44edae81b05022def55167b4559670d9a1666d8723682e46cb";
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
    '';
  };
}
