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
    # Check if Rust is installed, if not install it
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    fi

    # Check if OpenSSL 1.1 is installed
    if ! (command -v openssl &> /dev/null && openssl version | grep -q "OpenSSL 1.1"); then

        # Check the architecture and install OpenSSL 1.1 if needed
        if [[ $(uname -m) == "arm64" ]]; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # MacOS M1 installation
                if ! command -v brew &> /dev/null; then
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                fi

                export PATH="/opt/homebrew/bin:$PATH"

                brew install openssl@1.1

            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                # Check for Debian-based system
                if grep -qi 'debian' /etc/os-release; then
                    # Ubuntu ARM installation

                    apt update && apt install curl -y

                    curl -s https://packagecloud.io/install/repositories/wasmcloud/core/script.deb.sh | bash

                    apt install wash

                    curl -fLO http://ftp.us.debian.org/debian/pool/main/o/openssl/libssl1.1_1.1.1n-0+deb11u4_arm64.deb

                    dpkg -i libssl1.1_1.1.1n-0+deb11u4_arm64.deb

                else
                    echo "This script is designed for Debian-based systems only."
                    exit 1
                fi

            else
                echo "Unsupported system type."
                exit 1
            fi

        else
            echo "This script is designed for arm64 systems only."
            exit 1
        fi

    fi

    # Check if cosmo is installed, if not install it
    if ! command -v cosmo &> /dev/null; then
        bash -c "$(curl -fsSL https://cosmonic.sh/install.sh)"

        # Get the current shell name
        current_shell="$(basename "$SHELL")"

        # Update the corresponding configuration file based on the current shell
        if [[ "$current_shell" == "bash" ]]; then
            echo "export PATH=\"/Users/test/.cosmo/bin:\${PATH}\"" >> "${HOME}/.bashrc" && source "${HOME}/.bashrc"
        elif [[ "$current_shell" == "zsh" ]]; then
            echo "export PATH=\"/Users/test/.cosmo/bin:\${PATH}\"" >> "${HOME}/.zshrc" && source "${HOME}/.zshrc"
        else
            echo "Unsupported shell: $current_shell"
            exit 1
        fi

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
