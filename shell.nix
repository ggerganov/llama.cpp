{ pkgs ? import <nixpkgs> { overlays = [ (import ./vespa-cli-overlay.nix) ]; } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    just
    cargo
    tree
    poetry
    openssl_1_1
    vespa-cli
  ];
  shellHook = ''
    # Check if cosmo is installed, if not install it
    if ! command -v cosmo &> /dev/null; then
      bash -c "$(curl -fsSL https://cosmonic.sh/install.sh)"
    fi

    cat <<'EOF'
               .-.
                  `-'
    ___          LoRA
  .´   `'.     _...._
 :  LLAMA  :  .'      '.
  '._____.' /`(o)    (o)`\
  _|_______|/  :      :  \
[_____________/ '------'  \
  /  o                    /
  `"`"|"`"`"`"`"`"`"`""=""===""`
EOF

    echo "gm gm ⟁"
  '';
}
