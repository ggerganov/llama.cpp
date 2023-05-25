scripts_dir := "scripts"

# Check if Nix is installed, if not install it based on the platform
install-nix:
    . {{scripts_dir}}/install_nix.sh; \

# Check if Rust is installed, if not install it
install-rust:
    @if ! command -v rustc &> /dev/null; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; \
        . {{scripts_dir}}/update_source.sh; \
    fi

# Check if wasm32-unknown-unknown target is installed, if not install it
install-wasm-target:
    @if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then \
        rustup target add wasm32-unknown-unknown; \
        . {{scripts_dir}}/update_source.sh; \
    fi

# Check if OpenSSL 1.1 is installed
install-openssl:
    @if ! (command -v openssl &> /dev/null && openssl version | grep -q "OpenSSL 1.1"); then \
        . {{scripts_dir}}/install_openssl.sh; \
        . {{scripts_dir}}/update_source.sh; \
    fi

# Check if cosmo is installed, if not install it
install-cosmo:
    @if ! command -v cosmo &> /dev/null; then \
        bash -c "$(curl -fsSL https://cosmonic.sh/install.sh)"; \
        . {{scripts_dir}}/update_path.sh; \
        . {{scripts_dir}}/update_source.sh; \
    fi

# Print a message to restart the shell
restart-shell-message:
    @echo
    @echo "\033[1;33mPlease restart your shell to refresh the source before launching Cosmonic.\033[0m"

all: install-nix install-rust install-wasm-target install-openssl install-cosmo restart-shell-message
