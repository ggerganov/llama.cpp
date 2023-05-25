#!/bin/bash

# Get the current shell name
current_shell="$(basename "$SHELL")"

# Update the corresponding configuration file based on the current shell
if [[ "$current_shell" == "bash" ]]; then
    cat >> "${HOME}/.bashrc" <<EOF
export PATH="/Users/test/.cosmo/bin:\${PATH}"
EOF
    source "${HOME}/.bashrc"
elif [[ "$current_shell" == "zsh" ]]; then
    cat >> "${HOME}/.zshrc" <<EOF
export PATH="/Users/test/.cosmo/bin:\${PATH}"
EOF
    source "${HOME}/.zshrc"
else
    echo "Unsupported shell: $current_shell"
    exit 1
fi
