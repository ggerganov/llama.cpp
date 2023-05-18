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
