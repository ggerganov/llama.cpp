scripts_dir := "scripts"

# Check if Nix is installed, if not install it based on the platform
install-nix:
    . {{scripts_dir}}/install_nix.sh; \

# Check if Rust is installed, if not install it
install-rust:
    @if ! command -v rustc &> /dev/null; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh; \
        . {{scripts_dir}}/update_path.sh; \
    fi

# Check if wasm32-unknown-unknown target is installed, if not install it
install-wasm-target:
    @if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then \
        rustup target add wasm32-unknown-unknown; \
        . {{scripts_dir}}/update_path.sh; \
    fi

# Check if OpenSSL 1.1 is installed
install-openssl:
    @if ! (command -v openssl &> /dev/null && openssl version | grep -q "OpenSSL 1.1"); then \
        . {{scripts_dir}}/install_openssl.sh; \
        . {{scripts_dir}}/update_path.sh; \
    fi

# Check if cosmo is installed, if not install it
install-cosmo:
    @if ! command -v cosmo &> /dev/null; then \
        bash -c "$(curl -fsSL https://cosmonic.sh/install.sh)"; \
        current_shell=$$(basename "$$SHELL"); \
        if [ "$$current_shell" == "bash" ]; then \
            echo "export PATH=\"/Users/nexus/.cosmo/bin:\$$${PATH}\"" >> "${HOME}/.bashrc" && source "${HOME}/.bashrc"; \
        elif [ "$$current_shell" == "zsh" ]; then \
            echo "export PATH=\"/Users/nexus/.cosmo/bin:\$$${PATH}\"" >> "${HOME}/.zshrc" && source "${HOME}/.zshrc"; \
        else \
            echo "Unsupported shell: $$current_shell"; \
        fi; \
    fi

all: install-nix install-rust install-wasm-target install-openssl install-cosmo
