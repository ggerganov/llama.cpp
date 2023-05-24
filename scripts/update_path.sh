#!/bin/bash

# Get the current shell name
current_shell="$(basename ${builtins.getEnv "SHELL"})"

# Update the corresponding configuration file based on the current shell
if [[ "$current_shell" == "bash" ]]; then
    cat >> "${builtins.getEnv "HOME"}/.bashrc" <<EOF
export PATH="/Users/test/.cosmo/bin:\${builtins.getEnv "PATH"}"
EOF
    source "${builtins.getEnv "HOME"}/.bashrc"
elif [[ "$current_shell" == "zsh" ]]; then
    cat >> "${builtins.getEnv "HOME"}/.zshrc" <<EOF
export PATH="/Users/test/.cosmo/bin:\${builtins.getEnv "PATH"}"
EOF
    source "${builtins.getEnv "HOME"}/.zshrc"
else
    echo "Unsupported shell: $current_shell"
    exit 1
fi
