{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    just
    cargo
    tree
    poetry
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
